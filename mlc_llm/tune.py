"""Script for tuning models."""
from mlc_llm import core

def main():
    """Main method for tuning model from command line."""
    empty_args = core.convert_build_args_to_argparser()  # Create new ArgumentParser
    parsed_args = empty_args.parse_args()  # Parse through command line
    # Post processing of arguments
    parsed_args = core._parse_args(parsed_args)  # pylint: disable=protected-access
    tasks = core.extract_matmul_tasks_from_model(parsed_args)
    print("TASKS: ", len(tasks))
    for t in tasks:
        print(t.task_name)
        print(t.mod)

if __name__ == "__main__":
    main()
