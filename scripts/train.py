"""Distributed training entry point; the workflow selects the architecture trainer."""
from llamafactory.train.tuner import run_exp

if __name__ == "__main__":
    run_exp()
