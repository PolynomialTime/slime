import json
import logging
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.local_rm.reward_eval import reward_eval
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.misc import load_function, should_run_periodic_action

logger = logging.getLogger(__name__)


def add_irl_pipeline_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--reward-model-dir",
        type=str,
        default="reward_model",
        help="Directory to save and load local reward model checkpoints.",
    )
    parser.add_argument(
        "--reward-model-init",
        type=str,
        default=None,
        help="HF model path used to initialize reward model backbone. Defaults to --hf-checkpoint.",
    )
    parser.add_argument(
        "--reward-demo-path",
        type=str,
        default=None,
        help="Path to demo jsonl (prompt+answer) used to update reward model.",
    )
    parser.add_argument(
        "--reward-demo-prompt-key",
        type=str,
        default="prompt",
        help="JSON key for demo prompt.",
    )
    parser.add_argument(
        "--reward-demo-answer-key",
        type=str,
        default="answer",
        help="JSON key for demo answer.",
    )
    parser.add_argument(
        "--reward-update-epochs",
        type=int,
        default=1,
        help="Number of epochs for each reward update.",
    )
    parser.add_argument(
        "--reward-update-batch-size",
        type=int,
        default=8,
        help="Batch size for reward update.",
    )
    parser.add_argument(
        "--reward-update-lr",
        type=float,
        default=1e-5,
        help="Learning rate for reward update.",
    )
    parser.add_argument(
        "--c-coef-init",
        type=float,
        default=1.0,
        help="Initial coefficient for epsilon penalty.",
    )
    parser.add_argument(
        "--c-coef-min",
        type=float,
        default=0.1,
        help="Minimum coefficient for epsilon penalty.",
    )
    parser.add_argument(
        "--c-coef-max",
        type=float,
        default=10.0,
        help="Maximum coefficient for epsilon penalty.",
    )
    parser.add_argument(
        "--coef-scale-up",
        type=float,
        default=1.2,
        help="Scale up factor for c_coef when epsilon is too large.",
    )
    parser.add_argument(
        "--coef-scale-down",
        type=float,
        default=0.8,
        help="Scale down factor for c_coef when epsilon is too small.",
    )
    parser.add_argument(
        "--target-reward-l2-norm",
        type=float,
        default=5.0,
        help="Target L2 norm for reward delta (epsilon).",
    )
    parser.add_argument(
        "--reward-update-interval",
        type=int,
        default=1,
        help="Update reward model every N rollouts (1 means every rollout).",
    )
    parser.add_argument(
        "--reward-update-launcher",
        type=str,
        default="direct",
        choices=["direct", "accelerate"],
        help="How to launch reward update (direct in-process or accelerate).",
    )
    parser.add_argument(
        "--reward-update-accelerate-config",
        type=str,
        default=None,
        help="Accelerate config file path for reward update.",
    )
    parser.add_argument(
        "--reward-update-accelerate-num-proc",
        type=int,
        default=None,
        help="Number of processes for accelerate reward update.",
    )
    parser.add_argument(
        "--reward-update-rollout-window",
        type=int,
        default=1,
        help="Number of recent rollout files to aggregate for each reward update.",
    )
    parser.add_argument(
        "--reward-eval-path",
        type=str,
        default=None,
        help="Path to reward eval jsonl (prompt+chosen+rejected). Defaults to reward_demo_path if not set.",
    )
    parser.add_argument(
        "--reward-eval-prompt-key",
        type=str,
        default=None,
        help="JSON key for reward eval prompt. Defaults to reward_demo_prompt_key if not set.",
    )
    parser.add_argument(
        "--reward-eval-chosen-key",
        type=str,
        default="chosen",
        help="JSON key for reward eval chosen answer.",
    )
    parser.add_argument(
        "--reward-eval-rejected-key",
        type=str,
        default="rejected",
        help="JSON key for reward eval rejected answer.",
    )
    parser.add_argument(
        "--reward-eval-batch-size",
        type=int,
        default=None,
        help="Batch size for reward eval. Defaults to reward_update_batch_size if not set.",
    )
    parser.add_argument(
        "--reward-eval-max-samples",
        type=int,
        default=None,
        help="Max number of eval samples to score. If not set, evaluate all.",
    )
    parser.add_argument(
        "--reward-update-fn-path",
        type=str,
        default="slime.local_rm.update_reward.update_reward",
        help=(
            "Python path to reward update function. Signature: "
            "def update_reward(args, rollout_id, rollout_path) -> None"
        ),
    )
    return parser


def _maybe_set_default_debug_path(args) -> None:
    if args.save_debug_rollout_data is None:
        args.save_debug_rollout_data = "debug_rollout/rollout_{rollout_id}.pt"
        logger.info("save_debug_rollout_data not set, defaulting to %s", args.save_debug_rollout_data)


def _call_reward_update(args, rollout_id: int) -> bool:
    if args.reward_update_fn_path is None:
        return False
    if args.reward_update_interval is None or args.reward_update_interval <= 0:
        return False
    if rollout_id % args.reward_update_interval != 0:
        return False

    rollout_path = args.save_debug_rollout_data.format(rollout_id=rollout_id)
    if args.reward_update_launcher == "accelerate":
        reward_dir = Path(args.reward_model_dir)
        reward_dir.mkdir(parents=True, exist_ok=True)
        args_json_path = reward_dir / "reward_update_args.json"
        reward_args = {
            "hf_checkpoint": args.hf_checkpoint,
            "reward_model_dir": args.reward_model_dir,
            "reward_model_init": args.reward_model_init,
            "reward_demo_path": args.reward_demo_path,
            "reward_demo_prompt_key": args.reward_demo_prompt_key,
            "reward_demo_answer_key": args.reward_demo_answer_key,
            "reward_update_epochs": args.reward_update_epochs,
            "reward_update_batch_size": args.reward_update_batch_size,
            "reward_update_lr": args.reward_update_lr,
            "c_coef_init": args.c_coef_init,
            "c_coef_min": args.c_coef_min,
            "c_coef_max": args.c_coef_max,
            "coef_scale_up": args.coef_scale_up,
            "coef_scale_down": args.coef_scale_down,
            "target_reward_l2_norm": args.target_reward_l2_norm,
            "apply_chat_template": args.apply_chat_template,
            "apply_chat_template_kwargs": args.apply_chat_template_kwargs,
            "save_debug_rollout_data": args.save_debug_rollout_data,
            "reward_update_rollout_window": args.reward_update_rollout_window,
        }
        args_json_path.write_text(json.dumps(reward_args, ensure_ascii=False, indent=2), encoding="utf-8")
        cmd = ["accelerate", "launch"]
        if args.reward_update_accelerate_config:
            cmd += ["--config_file", args.reward_update_accelerate_config]
        if args.reward_update_accelerate_num_proc:
            cmd += ["--num_processes", str(args.reward_update_accelerate_num_proc)]
        cmd += [
            "-m",
            "slime.local_rm.update_reward_accel",
            "--args-json",
            str(args_json_path),
            "--rollout-id",
            str(rollout_id),
            "--rollout-path",
            rollout_path,
        ]
        logger.info("Launching reward update via accelerate: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        update_fn = load_function(args.reward_update_fn_path)
        logger.info("Updating reward model using %s on %s", args.reward_update_fn_path, rollout_path)
        update_fn(args, rollout_id, rollout_path)
    return True


def train(args) -> None:
    configure_logger()
    pgs = create_placement_groups(args)
    init_tracking(args)

    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train():
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            actor_model.clear_memory()

    def save(rollout_id: int):
        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    _maybe_set_default_debug_path(args)

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        did_reward_update = _call_reward_update(args, rollout_id)
        if did_reward_update:
            reward_eval(args, rollout_id)

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(rollout_id)

        offload_train()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args(add_custom_arguments=add_irl_pipeline_arguments)
    train(args)
