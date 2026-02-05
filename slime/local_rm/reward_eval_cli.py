import argparse
import json
from types import SimpleNamespace

from .reward_eval import reward_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--args-json", type=str, required=True)
    parser.add_argument("--rollout-id", type=int, required=True)
    return parser.parse_args()


def main():
    cli = parse_args()
    with open(cli.args_json, encoding="utf-8") as f:
        cfg_dict = json.load(f)
    args = SimpleNamespace(**cfg_dict)
    reward_eval(args, cli.rollout_id)


if __name__ == "__main__":
    main()
