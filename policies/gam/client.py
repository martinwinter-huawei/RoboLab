# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import numpy as np
import torch

from openpi_client import websocket_client_policy
from robolab.eval.base_client import InferenceClient

logger = logging.getLogger(__name__)


class GamClient(InferenceClient):
    """Inference client for Geometric-Action-Model (GAM).

    Connects to a GAM policy server over the OpenPI WebSocket protocol.
    Expects 7-DoF relative end-effector action chunks:
      [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """

    OPEN_LOOP_HORIZON = 8

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        open_loop_horizon: int = OPEN_LOOP_HORIZON,
    ):
        super().__init__()
        self._remote_host = remote_host
        self._remote_port = remote_port
        self.open_loop_horizon = open_loop_horizon

        display = f"{self._remote_host}:{self._remote_port}"
        print(f"[{self.__class__.__name__}] Awaiting server on {display} to be ready...")
        self.client = self._connect()
        print(f"[{self.__class__.__name__}] Connected to {display}.")

    def _connect(self) -> websocket_client_policy.WebsocketClientPolicy:
        return websocket_client_policy.WebsocketClientPolicy(self._remote_host, self._remote_port)

    def _infer_with_retry(self, request: dict, max_retries: int = 3) -> dict:
        import websockets.exceptions

        for attempt in range(max_retries):
            try:
                return self.client.infer(request)
            except (
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK,
                OSError,
            ) as e:
                if attempt + 1 >= max_retries:
                    raise
                logger.warning(
                    "[%s] Connection lost (%s), reconnecting (attempt %d/%d)...",
                    self.__class__.__name__,
                    e,
                    attempt + 1,
                    max_retries,
                )
                self.client = self._connect()
                self._chunks.clear()
                self._counters.clear()

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        """Extract camera observations and proprioception from RoboLab's raw observation dict."""
        # Exterior view (maps to agentview)
        left_image = raw_obs["image_obs"]["over_shoulder_left_camera"][env_id].cpu().numpy()

        # Wrist-mounted camera (maps to robot0_eye_in_hand)
        wrist_image = raw_obs["image_obs"]["wrist_cam"][env_id].cpu().numpy()

        # Proprioception
        joint_pos = raw_obs["proprio_obs"]["arm_joint_pos"][env_id].cpu().numpy()
        gripper_pos = raw_obs["proprio_obs"]["gripper_pos"][env_id].cpu().numpy()

        eef_pos = (
            raw_obs["proprio_obs"]["eef_pos"][env_id].cpu().numpy()
            if "eef_pos" in raw_obs["proprio_obs"]
            else None
        )
        eef_quat = (
            raw_obs["proprio_obs"]["eef_quat"][env_id].cpu().numpy()
            if "eef_quat" in raw_obs["proprio_obs"]
            else None
        )

        return {
            "image": left_image,
            "wrist_image": wrist_image,
            "joint_pos": joint_pos,
            "gripper_pos": gripper_pos,
            "eef_pos": eef_pos,
            "eef_quat": eef_quat,
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        """Format payload for the GAM policy server."""
        payload = {
            "image": extracted_obs["image"],
            "wrist_image": extracted_obs["wrist_image"],
            "joint_pos": extracted_obs["joint_pos"],
            "gripper_pos": extracted_obs["gripper_pos"],
            "prompt": instruction,
        }
        if extracted_obs["eef_pos"] is not None:
            payload["eef_pos"] = extracted_obs["eef_pos"]
        if extracted_obs["eef_quat"] is not None:
            payload["eef_quat"] = extracted_obs["eef_quat"]
        return payload

    def _query_server(self, request: dict) -> dict:
        return self._infer_with_retry(request)

    def _unpack_response(self, response: dict) -> np.ndarray:
        """Expects (horizon, 7) array: [dx, dy, dz, droll, dpitch, dyaw, gripper]."""
        actions = np.asarray(response["action"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]
        return actions

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Ensure chunk has valid shape and gripper values."""
        chunk = chunk.copy()
        if chunk.shape[-1] >= 7:
            gripper = chunk[..., 6]
            chunk[..., 6] = np.where(gripper > 0.0, 1.0, -1.0)
        return chunk

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        """Horizontal concatenation of exterior and wrist cameras for visualizer."""
        agentview = extracted_obs["image"]
        wrist = extracted_obs["wrist_image"]
        if agentview.shape[0] != wrist.shape[0]:
            from PIL import Image
            wrist = np.array(Image.fromarray(wrist).resize(
                (int(wrist.shape[1] * agentview.shape[0] / wrist.shape[0]), agentview.shape[0])
            ))
        return np.concatenate((agentview, wrist), axis=1)
