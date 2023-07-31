# pylint: disable=missing-docstring,invalid-name
import argparse
import json
import os
import shutil
from typing import Any, Dict, List, Optional, Set, Tuple

import tvm
from tvm import meta_schedule as ms
from tvm import relax

from .quantization import quantization_schemes
from .relax_model import param_manager


supported_model_types = set(
    ["llama", "gpt_neox", "gpt_bigcode", "minigpt", "moss", "rwkv", "gptj"]
)


def argparse_postproc_common(args: argparse.Namespace) -> None:
    if hasattr(args, "device_name"):
        if args.device_name == "auto":
            if tvm.cuda().exist:
                args.device_name = "cuda"
            elif tvm.metal().exist:
                args.device_name = "metal"
            elif tvm.vulkan().exist:
                args.device_name = "vulkan"
            elif tvm.opencl().exist:
                args.device_name = "opencl"
            else:
                raise ValueError("Cannot auto deduce device-name, please set it")

    model_category_override = {
        "moss-moon-003-sft": "gptj",
        "moss-moon-003-base": "gptj",
        "rwkv-": "rwkv",
        "minigpt": "minigpt",
    }
    try:
        with open(
            os.path.join(args.model_path, "config.json"), encoding="utf-8"
        ) as i_f:
            config = json.load(i_f)
            args.model_category = config["model_type"]
    except Exception:
        args.model_category = ""
    model = args.model.lower()
    for prefix, override_category in model_category_override.items():
        if model.startswith(prefix):
            args.model_category = override_category
            break
    assert args.model_category is not None

    model_conv_templates = {
        "llama-2": "llama-2",
        "vicuna-": "vicuna_v1.1",
        "dolly-": "dolly",
        "stablelm-": "stablelm",
        "redpajama-": "redpajama_chat",
        "minigpt": "minigpt",
        "moss-moon-003-sft": "moss",
        "moss-moon-003-base": "LM",
        "gpt-j-": "LM",
        "open_llama": "LM",
        "rwkv-": "rwkv",
        "gorilla-": "gorilla",
        "guanaco": "guanaco",
        "starcoder": "code_gpt",
        "wizardcoder-": "code_gpt",
        "wizardlm-": "wizardlm",
        "gpt_bigcode-santacoder": "code_gpt",
    }

    for prefix, conv_template in model_conv_templates.items():
        if model.startswith(prefix):
            args.conv_template = conv_template
            break
    else:
        args.conv_template = f"{args.model_category}_default"

    if args.quantization not in quantization_schemes:
        raise ValueError(f'Quantization "{args.quantization}" is not supported.')
    args.quantization = quantization_schemes[args.quantization]


def extract_transform_mod(mod: tvm.IRModule, func_name: str) -> tvm.IRModule:
    mod_transform = tvm.IRModule({tvm.ir.GlobalVar(func_name): mod[func_name]})
    for gv in mod.functions:
        func = mod[gv]
        if isinstance(func, tvm.tir.PrimFunc):
            mod_transform[gv] = func
    mod_transform = relax.transform.DeadCodeElimination([func_name])(mod_transform)
    return mod_transform


def split_transform_deploy_mod(
    mod: tvm.IRModule, model_names: List[str]
) -> Tuple[tvm.IRModule, tvm.IRModule]:
    mod_transform = tvm.IRModule()
    mod_deploy = tvm.IRModule()
    transform_func_name = "transform_params"
    transform_func_names=[]

    for gv in mod.functions:
        func = mod[gv]
        if isinstance(func, tvm.tir.PrimFunc):
            mod_transform[gv] = func
            mod_deploy[gv] = func
        elif transform_func_name in gv.name_hint:
            mod_transform[gv] = func
            transform_func_names.append(gv.name_hint)
        else:
            mod_deploy[gv] = func

    mod_transform = relax.transform.DeadCodeElimination(transform_func_names)(mod_transform)
    mod_deploy = relax.transform.DeadCodeElimination(model_names)(mod_deploy)

    return mod_transform, mod_deploy


def transform_params(
    mod_transform: tvm.IRModule,
    transform_func_name: str,
    param_manager: param_manager.ParamManager,
    model_params: List[Optional[tvm.nd.NDArray]],
) -> List[tvm.nd.NDArray]:
    target = detect_local_target()
    print(f"Automatically using target for weight quantization: {target}")
    device = tvm.device(target.kind.default_keys[0])
    device_cpu = tvm.cpu()

    loaded_params: List[tvm.nd.NDArray] = []
    loaded_idx_set: Set[int] = set()
    loaded_torch_bins: Set[str] = set()
    cached_relax_params: Dict[int, tvm.nd.NDArray] = {}
    cached_torch_params: Dict[str, Any] = {}

    get_item, set_item = param_manager.get_param_loading_functions(
        model_params,
        loaded_params,
        loaded_idx_set,
        loaded_torch_bins,
        cached_relax_params,
        cached_torch_params,
        device,
        device_cpu,
    )
    tvm.register_func(func_name="get_item", f=get_item, override=True)
    tvm.register_func(func_name="set_item", f=set_item, override=True)

    if target.kind.name != "llvm":
        with tvm.target.Target(target):
            mod_transform = tvm.tir.transform.DefaultGPUSchedule()(mod_transform)

    ex = relax.build(mod_transform, target=target)
    vm = relax.vm.VirtualMachine(ex, device)
    print("Start computing and quantizing weights... This may take a while.")
    vm[transform_func_name]()
    print("Finish computing and quantizing weights.")
    return loaded_params


def save_params(params: List[tvm.nd.NDArray], artifact_path: str) -> None:
    from tvm.contrib import tvmjs  # pylint: disable=import-outside-toplevel

    meta_data = {}
    param_dict = {}
    meta_data["ParamSize"] = len(params)
    total_size = 0.0
    for i, nd in enumerate(params):
        param_dict[f"param_{i}"] = nd
        np_nd = nd.numpy()
        total_size += np_nd.size * np_nd.dtype.itemsize
    total_size = total_size / 1024.0 / 1024.0 / 1024.0
    print(f"Total param size: {total_size} GB")
    tvmjs.dump_ndarray_cache(
        param_dict, f"{artifact_path}", meta_data=meta_data, encode_format="raw"
    )


def load_params(artifact_path: str, device) -> List[tvm.nd.NDArray]:
    from tvm.contrib import tvmjs  # pylint: disable=import-outside-toplevel

    params, meta = tvmjs.load_ndarray_cache(f"{artifact_path}", device)
    plist = []
    size = meta["ParamSize"]
    for i in range(size):
        plist.append(params[f"param_{i}"])
    return plist


def build_model_from_log(relax_mod, target, log_dir):
    db = ms.database.create(work_dir=log_dir)
    with target, db, tvm.transform.PassContext(opt_level=3):
        relax_mod = relax.transform.MetaScheduleApplyDatabase()(relax_mod)
    return relax_mod


def split_static_dynamic_tir(mod: tvm.IRModule):
    def _is_static_shape_buffer(buffer: tvm.tir.Buffer):
        for dim in buffer.shape:
            if not isinstance(dim, tvm.tir.IntImm):
                return False
        return True

    def _is_static_shape_func(func: tvm.tir.PrimFunc):
        for buffer in func.buffer_map.values():
            if not _is_static_shape_buffer(buffer):
                return False
        return True

    mod_dynamic = {}
    mod_static = {}
    for k, v in mod.functions.items():
        if isinstance(v, tvm.tir.PrimFunc):
            if _is_static_shape_func(v):
                mod_static[k] = v
            else:
                mod_dynamic[k] = v
    mod_static = tvm.IRModule(mod_static)
    mod_dynamic = tvm.IRModule(mod_dynamic)
    return mod_static, mod_dynamic


def copy_tokenizer(args: argparse.Namespace) -> None:
    dst_dir = os.path.join(args.artifact_path, "params")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for filename in os.listdir(args.model_path):
        if filename in [
            "tokenizer.model",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "tokenizer_config.json",
        ]:
            shutil.copy(os.path.join(args.model_path, filename), dst_dir)


def get_tokenizer_files(path) -> List[str]:
    tokenizer_set = {
        "tokenizer.model",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
    }
    return [x for x in os.listdir(path) if x in tokenizer_set]


def get_database(db_paths: str) -> ms.Database:
    db = ms.database.MemoryDatabase()  # pylint: disable=invalid-name
    for db_path in db_paths:
        model_db = ms.database.create(kind="json", work_dir=db_path)
        for record in model_db.get_all_tuning_records():
            db.commit_workload(record.workload.mod)
            db.commit_tuning_record(record)
    return db


def _detect_local_metal_host():
    target_triple = tvm._ffi.get_global_func(
        "tvm.codegen.llvm.GetDefaultTargetTriple"
    )()
    process_triple = tvm._ffi.get_global_func("tvm.codegen.llvm.GetProcessTriple")()
    host_cpu = tvm._ffi.get_global_func("tvm.codegen.llvm.GetHostCPUName")()
    print(
        f"Host CPU dection:\n  Target triple: {target_triple}\n  Process triple: {process_triple}\n  Host CPU: {host_cpu}"
    )
    if target_triple.startswith("x86_64-"):
        return tvm.target.Target(
            {
                "kind": "llvm",
                "mtriple": "x86_64-apple-macos",
                "mcpu": host_cpu,
            }
        )
    # should start with "arm64-"
    return tvm.target.Target(
        {
            "kind": "llvm",
            "mtriple": "arm64-apple-macos",
            "mcpu": host_cpu,
        }
    )


def _detect_local_metal():
    dev = tvm.metal()
    if not dev.exist:
        return None

    return tvm.target.Target(
        {
            "kind": "metal",
            "max_shared_memory_per_block": 32768,
            "max_threads_per_block": dev.max_threads_per_block,
            "thread_warp_size": 32,
        },
        host=_detect_local_metal_host(),
    )


def _detect_local_cuda():
    dev = tvm.cuda()
    if not dev.exist:
        return None
    return tvm.target.Target(
        {
            "kind": "cuda",
            "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            "max_threads_per_block": dev.max_threads_per_block,
            "thread_warp_size": dev.warp_size,
            "registers_per_block": 65536,
            "arch": "sm_" + tvm.cuda().compute_version.replace(".", ""),
        }
    )


def _detect_local_vulkan():
    dev = tvm.vulkan()
    if not dev.exist:
        return None
    return tvm.target.Target(
        {
            "kind": "vulkan",
            "max_threads_per_block": dev.max_threads_per_block,
            "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            "thread_warp_size": dev.warp_size,
            "supports_float16": 1,
            "supports_int16": 1,
            "supports_int8": 1,
            "supports_16bit_buffer": 1,
        }
    )


def _detect_local_opencl():
    dev = tvm.opencl()
    if not dev.exist:
        return None
    return tvm.target.Target("opencl")


def detect_local_target():
    for method in [
        _detect_local_metal,
        _detect_local_cuda,
        _detect_local_vulkan,
        _detect_local_opencl,
    ]:
        target = method()
        if target is not None:
            return target

    print("Failed to detect local GPU, falling back to CPU as a target")
    return tvm.target.Target("llvm")


def parse_target(args: argparse.Namespace) -> None:
    if not hasattr(args, "target"):
        return
    if args.target == "auto":
        target = detect_local_target()
        if target.host is None:
            target = tvm.target.Target(
                target,
                host="llvm",  # TODO: detect host CPU
            )
        args.target = target
        args.target_kind = args.target.kind.default_keys[0]
    elif args.target == "metal":
        target = _detect_local_metal()
        if target is None:
            print("Cannot detect local Apple Metal GPU target! Falling back...")
            target = tvm.target.Target(
                tvm.target.Target(
                    {
                        "kind": "metal",
                        "max_threads_per_block": 256,
                        "max_shared_memory_per_block": 32768,
                        "thread_warp_size": 1,
                    }
                ),
                host=_detect_local_metal_host(),
            )
        args.target = target
        args.target_kind = args.target.kind.default_keys[0]
    elif args.target == "metal_x86_64":
        from tvm.contrib import xcode  # pylint: disable=import-outside-toplevel

        args.target = tvm.target.Target(
            tvm.target.Target(
                {
                    "kind": "metal",
                    "max_threads_per_block": 256,
                    "max_shared_memory_per_block": 32768,
                    "thread_warp_size": 1,
                }
            ),
            host="llvm -mtriple=x86_64-apple-darwin",
        )
        args.target_kind = "metal_x86_64"
        args.export_kwargs = {
            "fcompile": xcode.create_dylib,
            "sdk": "macosx",
            "arch": "x86_64",
        }
        args.lib_format = "dylib"
    elif args.target in ["iphone", "iphone-dylib", "iphone-tar"]:
        from tvm.contrib import tar, xcode  # pylint: disable=import-outside-toplevel

        if args.target == "iphone-dylib":
            args.export_kwargs = {
                "fcompile": xcode.create_dylib,
                "sdk": "iphoneos",
                "arch": "arm64",
            }
            args.lib_format = "dylib"
        else:
            args.export_kwargs = {"fcompile": tar.tar}
            args.lib_format = "tar"
            args.system_lib = True
            args.system_lib_prefix = f"{args.model}_{args.quantization}_".replace(
                "-", "_"
            )

        @tvm.register_func("tvm_callback_metal_compile")
        def compile_metal(src, target):
            if target.libs:
                return xcode.compile_metal(src, sdk=target.libs[0])
            return xcode.compile_metal(src)

        target = tvm.target.Target(
            tvm.target.Target(
                {
                    "kind": "metal",
                    "max_threads_per_block": 256,
                    "max_shared_memory_per_block": 32768,
                    "thread_warp_size": 1,
                    "libs": ["iphoneos"],
                }
            ),
            host="llvm -mtriple=arm64-apple-darwin",
        )
        args.target = target
        args.target_kind = "iphone"
    elif args.target == "vulkan":
        target = tvm.target.Target(
            tvm.target.Target(
                {
                    "kind": "vulkan",
                    "max_threads_per_block": 256,
                    "max_shared_memory_per_block": 32768,
                    "thread_warp_size": 1,
                    "supports_float16": 1,
                    "supports_int16": 1,
                    "supports_int8": 1,
                    "supports_8bit_buffer": 1,
                    "supports_16bit_buffer": 1,
                    "supports_storage_buffer_storage_class": 1,
                }
            ),
            host="llvm",
        )
        args.target = target
        args.target_kind = args.target.kind.default_keys[0]
    elif args.target == "opencl":
        target = tvm.target.Target(
            "opencl",
            host="llvm",
        )
        args.target = target
        args.target_kind = args.target.kind.default_keys[0]
    elif args.target == "webgpu":
        args.target = tvm.target.Target(
            "webgpu",
            host="llvm -mtriple=wasm32-unknown-unknown-wasm",
        )
        args.target_kind = "webgpu"
        args.lib_format = "wasm"
        args.system_lib = True
        if os.environ.get("TVM_HOME", "") == "":
            raise RuntimeError(
                "Please set TVM_HOME for webgpu build following scripts/prep_emcc_deps.sh"
            )
    elif args.target in ["android", "android-dylib"]:  # android-opencl
        from tvm.contrib import ndk, tar

        if args.target == "android-dylib":
            args.export_kwargs = {
                "fcompile": ndk.create_shared,
            }
            args.lib_format = "so"
        else:
            args.export_kwargs = {
                "fcompile": tar.tar,
            }
            args.lib_format = "tar"
            args.system_lib = True
            args.system_lib_prefix = f"{args.model}_{args.quantization}_".replace(
                "-", "_"
            )
        args.target = tvm.target.Target(
            "opencl",
            host="llvm -mtriple=aarch64-linux-android",  # TODO: Only support arm64 for now
        )
        args.target_kind = "android"
    else:
        args.target = tvm.target.Target(args.target, host="llvm")
        args.target_kind = args.target.kind.default_keys[0]

    # use mingw to cross compile windows
    if hasattr(args, "llvm_mingw") and args.llvm_mingw != "":
        from tvm.contrib.cc import (  # pylint: disable=import-outside-toplevel
            cross_compiler,
        )

        args.export_kwargs = {
            "fcompile": cross_compiler(
                os.path.join(args.llvm_mingw, "bin", "x86_64-w64-mingw32-clang++"),
                output_format="dll",
            ),
        }
        args.target = args.target.with_host("llvm -mtriple=x86_64-w64-windows-gnu")
        args.lib_format = "dll"

    print(f"Target configured: {args.target}")
