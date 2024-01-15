import os
import subprocess
import shutil
import argparse

def run_build(
    path_to_model: str,
    num_calib_samples: int,
    alpha: float,
    path_to_logs: str,
    path_to_save_model: str
) -> None:
    subprocess.run(
        f"""python build.py --model={path_to_model} \
                    --use-cache=0 \
                    --quantization=smq_q8i8f16_2 \
                    --num-calib-samples={num_calib_samples} \
                    --alpha={alpha} \
                    --max-seq-len=2048 \
                    --dataset=trivia_qa  \
                    --artifact-path={path_to_save_model} > {path_to_logs}/llama2-7b-smq-alpha_{alpha}-triviaqa-{num_calib_samples}.log""",
        shell=True,
        universal_newlines=True
    )

def run_benchmark(
    path_to_benchmark: str,
    path_to_model: str,
    path_to_logs: str,
    num_calib_samples: int,
    alpha: float
) -> None:
    work_dir = os.getcwd()
    os.chdir(path_to_benchmark)
    subprocess.run(
        f"""python3 {os.path.join(path_to_benchmark, 'main.py')} --model=mlc-llm --model_args="model_name='Llama2-7b-smq_q8i8f16_2',model_path='{path_to_model}/Llama2-7b-smq_q8i8f16_2'" \
        --task=gsm8k \
        --output_path={path_to_logs}/gsm8k-llama2-7b-smq_alpha_{alpha}-triviaqa-{num_calib_samples}.json \
        --no_cache \
        --num_fewshot=0 \
        --batch_size=1 \
        --write_out \
        --output_base_path={path_to_logs}""",
        shell=True
    )
    os.chdir(work_dir)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_model", type=str, required=True)
    parser.add_argument("--path_to_save_model", type=str, required=True)
    parser.add_argument("--path_to_benchmark", type=str, required=True)
    args = parser.parse_args()

    path_to_logs = os.path.join(os.getcwd(), "smq_alpha_results")
    if not os.path.exists(path_to_logs):
        os.mkdir(path_to_logs)

    for num_calib_samples in [1, 3, 5, 10, 25, 50, 100, 125, 150, 175, 200, 225, 250, 275, 300]:
        for alpha in range(1, 9):        
            alpha = round(alpha / 10, 2)
            print(f"Running Llama2-7b with SmoothQuant (alpha = {alpha}, num_calib_samples = {num_calib_samples})")
            os.makedirs(args.path_to_save_model)
            run_build(args.path_to_model, num_calib_samples, alpha, path_to_logs, args.path_to_save_model)
            run_benchmark(args.path_to_benchmark, args.path_to_save_model, path_to_logs, num_calib_samples, alpha)
            shutil.rmtree(args.path_to_save_model)
            print("Done")

if __name__ == "__main__":
    main()
