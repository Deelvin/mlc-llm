from pathlib import Path
import argparse, os
import tvm
from tvm import relax
from mlc_llm import utils
from mlc_serve.model.base import get_model_artifact_config
import torch
import numpy as np

def reproduce(args: argparse.Namespace):
    # load a model
    config = get_model_artifact_config(args.model_artifact_path)
    lib_path = os.path.join(config.model_artifact_path, config.library_name)
    ex = tvm.runtime.load_module(lib_path)
    dev = tvm.device("cuda", 0)
    vm = relax.VirtualMachine(ex, dev)
    params = utils.load_params(config.model_artifact_path, dev)
    init_cache_func = tvm.get_global_func("tvm.contrib.vllm.allocate_kv_cache")

    # load saved assets
    def to_ndarray_via_torch(arr, torch_dtype):
        return tvm.nd.from_dlpack(torch.tensor(arr, dtype=torch_dtype, device="cuda"))

    input_ids_np = np.load(args.dump_dir+"/input_ids.npy")
    positions_np = np.load(args.dump_dir+"/positions.npy")
    seq_lens_np = np.load(args.dump_dir+"/seq_lens.npy")
    slot_mapping_np = np.load(args.dump_dir+"/slot_mapping.npy")
    block_tables_np = np.load(args.dump_dir+"/block_tables.npy")

    input_ids_np = [29899]
    positions_np = [16 + 16 + 16]
    seq_lens_np = [16 + 16 + 16 + 1]
    slot_mapping_np = [123 * 16]
    # block_tables_np = [[104, 123]] # pass
    block_tables_np = [[47, 126, 50, 123]] # pass



    # input_ids_np = input_ids_np[-1:]
    # # # originally 29899
    # # # input_ids_np[-1] = 13
    # positions_np = positions_np[-1:]
    # positions_np[0] = positions_np[0] - 14 # !!!! it must be there and in seq_lens_np greater on 1 !!!!
    # seq_lens_np = seq_lens_np[-1:]
    # seq_lens_np[0] = seq_lens_np[0] - 14
    # slot_mapping_np = slot_mapping_np[-1:]
    # # # 1928, 1982]
    # # #slot_mapping_np[0] = 108 * 16 + 14
    # # # slot_mapping_np[0] = 1982
    # slot_mapping_np[0] = slot_mapping_np[0] - 14
    # # # slot_mapping_np[0] = 107 * 16
    # # slot_mapping_np[0] = 16
    # block_tables_np = block_tables_np[-1:]
    # # #[104, 105, 106, 107, 108, 120,   0,   0,   0,   0,   0,   0,   0,
    # # #[ 47, 126,  50, 123,   0,   0,   0,   0,   0,   0,   0,   
    # # # block_tables_np[0][3]=108
    # # block_tables_np[0][3]=1

    # input_ids_np = input_ids_np[-2:-1]
    # positions_np = positions_np[-2:-1]
    # seq_lens_np = seq_lens_np[-2:-1]
    # slot_mapping_np = slot_mapping_np[-2:-1]
    # block_tables_np = block_tables_np[-2:-1]
    # #[104, 105, 106, 107, 108, 120,   0,   0,   0,   0,   0,   0,   0,
    # #[ 47, 126,  50, 123,   0,   0,   0,   0,   0,   0,   0,   
    # block_tables_np[0][5]=123

    input_ids = to_ndarray_via_torch(input_ids_np, torch.int)
    positions = to_ndarray_via_torch(positions_np, torch.int)
    seq_lens = to_ndarray_via_torch(seq_lens_np, torch.int)
    slot_mapping = to_ndarray_via_torch(slot_mapping_np, torch.int)
    block_tables = to_ndarray_via_torch(block_tables_np, torch.int)

    # Andrey has limited num_blocks during inference by 128
    num_blocks = 128
    block_size: int = 16
    num_heads = (
        config.num_key_value_heads
            // config.num_shards
        )
    num_layers = config.num_hidden_layers
    head_size = (
            config.hidden_size
            // config.num_attention_heads
        )

    kv_cache = init_cache_func(
        head_size, num_layers, num_heads, block_size, num_blocks
    )

    for i in range(0,config.num_hidden_layers * 2 ): # * 2 -> for keys and values
        kv_cache[i].copyfrom(
            to_ndarray_via_torch(np.load(f"{args.dump_dir}/kv_cache{i}.npy"), torch.float16))

    # call a model
    out = vm.module["decode"](
        input_ids,
        positions,
        seq_lens,
        kv_cache,
        slot_mapping,
        block_tables,
        params,
    )

    # analyze out for NaN
    logits = out[0]
    plogits = torch.from_dlpack(logits)
    print(torch.sum(torch.isnan(plogits) | torch.isinf(plogits)))

    # load logits collected in server
    logits_orig = to_ndarray_via_torch(np.load(args.dump_dir+"/logits.npy"), torch.float16)
    plogits_orig = torch.from_dlpack(logits_orig)
    print(torch.sum(torch.isnan(plogits_orig) | torch.isinf(plogits_orig)))
    a = 0
    for i in range(64):
        a += torch.sum(torch.isnan(torch.from_dlpack(kv_cache[i])) | torch.isinf(torch.from_dlpack(kv_cache[i])))
    print(a)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-id", type=str, required=True)
    parser.add_argument("--artifact-path", type=str, default="./dist")
    parser.add_argument("--dump_dir", type=str, required=True)
    args = parser.parse_args()

    args.model_artifact_path = Path(os.path.join(args.artifact_path, args.local_id))

    if not os.path.exists(args.model_artifact_path):
        raise Exception(f"Invalid local id: {args.local_id}")

    reproduce(args)
