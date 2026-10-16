# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the Geometric-Action-Model (GAM) policy backend across registered tasks."""

import argparse
import sys
import traceback

import cv2  # Must import before isaaclab. Do not remove.
from isaaclab.app import AppLauncher

POLICY = "gam"

parser = argparse.ArgumentParser(description="Evaluate Geometric-Action-Model (GAM) policy backend.")
parser.add_argument(
    "--remote-host", default="localhost", help="Remote host for policy server (default: localhost)."
)
parser.add_argument(
    "--remote-port", default=8000, type=int, help="Remote port for policy server (default: 8000)."
)
parser.add_argument(
    "--horizon", default=8, type=int, help="Open-loop action horizon / chunk size (default: 8)."
)
parser.add_argument(
    "--jointpos", action="store_true", help="Use joint-position action space instead of relative EE differential IK."
)

from robolab.eval.runner import add_common_eval_args, run_evaluation

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from policies.gam.client import GamClient
from robolab.registrations.droid.camera_presets import WRIST_LEFT

if args_cli.jointpos:
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
    auto_register_droid_envs(task=args_cli.task, cameras=WRIST_LEFT)
else:
    from robolab.registrations.droid.auto_env_registrations_rel_ik import auto_register_droid_rel_ik_envs
    # Register tasks with Droid differential relative IK (dx, dy, dz, droll, dpitch, dyaw, gripper)
    auto_register_droid_rel_ik_envs(task=args_cli.task, cameras=WRIST_LEFT)


def make_client(args: argparse.Namespace) -> GamClient:
    return GamClient(
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        open_loop_horizon=args.horizon,
    )


def main() -> None:
    run_evaluation(args_cli, policy=POLICY, client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
