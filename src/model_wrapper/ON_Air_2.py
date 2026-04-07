import os
import json
import time
import math
import torch
import numpy as np
import airsim
import torch.nn.functional as F

# 引入 httpx 用于显式绑定代理
import httpx
from openai import OpenAI

from model_wrapper.base_model import BaseModelWrapper
from airsim_plugin.airsim_settings import AirsimActionSettings
from model_wrapper.Qwen_api_captions import generate_caption, encode_image
from src.common.param import args
from common.prompts import (
    fixed_system_prompt, fixed_user_prompt_template,
    unfixed_system_prompt, unfixed_user_prompt_template
)


class ONAir(BaseModelWrapper):
    def __init__(self, fixed, batch_size):
        super().__init__()
        self.fixed = fixed

        # ==============================================================
        # 核心修改 2：给 OpenAI 客户端显式挂载代理
        # (用字符串拼接方式防止 IDE 或聊天框自动将其转为 Markdown 超链接)
        # ==============================================================
        proxy_url = "http://" + "127.0.0.1:13849"
        custom_http_client = httpx.Client(proxy=proxy_url)

        self.gpt_client = OpenAI(
            api_key="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",
            http_client=custom_http_client,  # 强制走外网代理
            timeout=30.0
        )

        self.start_position = [[] for _ in range(batch_size)]
        self.start_yaw = [0 for _ in range(batch_size)]
        self.current_poses = [[] for _ in range(batch_size)]

    def prepare_inputs(self, episodes, fixed):
        inputs, user_prompts, images, depth_images = [], [], [], []

        for i in range(len(episodes)):
            for src in episodes[i][::-1]:
                if 'rgb' in src and 'depth' in src:
                    images.extend(src['rgb'])
                    depth_images.extend(src['depth'])
                    break

        b64_imgs = encode_image(images)
        captions = generate_caption(b64_imgs)
        depth_info = self.process_depth(depth_images)

        for i in range(len(episodes)):
            captions4 = captions[4 * i: 4 * i + 4]
            depth_info4 = depth_info[4 * i: 4 * i + 4]
            self.start_position[i] = episodes[i][-1]['start_position']

            raw_quat = episodes[i][-1]['start_quaternionr']
            quaternionr = airsim.Quaternionr(x_val=raw_quat[0], y_val=raw_quat[1], z_val=raw_quat[2], w_val=raw_quat[3])
            pitch, roll, yaw = airsim.to_eularian_angles(quaternionr)
            self.start_yaw[i] = math.degrees(yaw)

            pre_poses = self.process_poses(episodes[i][-1]['pre_poses'])
            self.current_poses[i] = [pre_poses[-1][0][0], pre_poses[-1][0][1], pre_poses[-1][0][2], pre_poses[-1][1]]

            x_min = int(math.floor(self.start_position[i][0] - 500000))
            x_max = int(math.ceil(self.start_position[i][0] + 500000))
            y_min = int(math.floor(self.start_position[i][1] - 500000))
            y_max = int(math.ceil(self.start_position[i][1] + 500000))

            step_num = episodes[i][-1].get('step', 0)
            move_distance = episodes[i][-1].get('move_distance', 0.0)
            AvgHeadingChange = episodes[i][-1].get('avg_heading_changes', 0.0)
            object_name = episodes[i][-1].get('object_name', '')
            object_size = episodes[i][-1].get('object_size', '')
            task_desc = episodes[i][-1].get('description', '')

            format_previous_position = "{\n" + "\n".join([f"    {p}," for p in pre_poses]) + "\n}"
            prompt_template = fixed_user_prompt_template if fixed else unfixed_user_prompt_template
            system_prompt = fixed_system_prompt if fixed else unfixed_system_prompt

            user_content = prompt_template.format(
                instruction=task_desc, description=task_desc, object_name=object_name, object_size=object_size,
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, captions=captions4, captions4=captions4,
                depth_info=depth_info4, pre_poses=pre_poses, format_previous_position=format_previous_position,
                step_num=step_num, move_distance=move_distance, AvgHeadingChange=AvgHeadingChange
            )

            inputs.append([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}])
            user_prompts.append(user_content)

        return inputs, user_prompts

    def unfixed_single_call(self, conversation):
        try:
            response = self.gpt_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation,
            )
            text = response.choices[0].message.content.strip().strip("[]`\"'")
            parts = [p.strip().strip("[]`\"'") for p in text.split(",")]
            action = parts[0].strip('\'"')
            try:
                value = float(parts[1])
            except:
                value = 500.0
            anomaly_found = (parts[2].lower() == 'true') if len(parts) > 2 else False
            return action, value, (action == 'stop' or anomaly_found)
        except Exception as e:
            print(f">>> [同步模式] 大模型调用失败: {e}")
            return 'forward', 500.0, False

    def fixed_single_call(self, conversation):
        try:
            response = self.gpt_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation,
            )
            text = response.choices[0].message.content.strip().strip("[]`\"'")
            parts = [p.strip().strip("[]`\"'") for p in text.split(",")]
            action = parts[0].strip('\'"')
            anomaly_found = parts[1].strip().lower() == 'true' if len(parts) > 1 else False
            return action, AirsimActionSettings.FORWARD_STEP_SIZE, (action == 'stop' or anomaly_found)
        except:
            return 'forward', AirsimActionSettings.FORWARD_STEP_SIZE, False

    def run(self, inputs, fixed, prompt_info_list=None):
        results = [self.fixed_single_call(conv) if fixed else self.unfixed_single_call(conv) for conv in inputs]

        actions = [r[0] for r in results]
        steps_size = [r[1] for r in results]
        predict_dones = [r[2] for r in results]

        new_actions, new_step_size = self.redirect_action(actions, steps_size, fixed)
        return list(new_actions), list(new_step_size), list(predict_dones)

    def process_depth(self, depth_images):
        depth_info = []
        for depth_image in depth_images:
            distance_image = np.array(depth_image) / 255.0 * 100
            x = torch.from_numpy(distance_image).unsqueeze(0).unsqueeze(0).float()
            y = -F.adaptive_max_pool2d(-x, (3, 3))
            depth_info.append(np.round(y.squeeze().cpu().numpy()).astype(int).tolist())
        return depth_info

    def process_poses(self, poses):
        pre_poses_xyzYaw = []
        for pose in poses:
            pos = pose['position']
            raw_quat = pose['quaternionr']
            quaternionr = airsim.Quaternionr(x_val=raw_quat[0], y_val=raw_quat[1], z_val=raw_quat[2], w_val=raw_quat[3])
            _, _, yaw = airsim.to_eularian_angles(quaternionr)
            pre_poses_xyzYaw.append(
                [(round(pos[0], 2), round(pos[1], 2), round(pos[2], 2)), round(math.degrees(yaw), 2)])
        return pre_poses_xyzYaw

    def redirect_action(self, actions, step_size, fixed):
        new_actions = [None] * len(actions)
        new_step_size = list(step_size)
        for i, action in enumerate(actions):
            new_actions[i] = action
            try:
                start = self.start_position[i]
                curr = self.current_poses[i]
                x_limit = [start[0] - 500000, start[0] + 500000]
                y_limit = [start[1] - 500000, start[1] + 500000]

                if action == 'forward':
                    rad = math.radians(curr[3])
                    next_x = curr[0] + math.cos(rad) * step_size[i]
                    next_y = curr[1] + math.sin(rad) * step_size[i]
                    if not (x_limit[0] < next_x < x_limit[1] and y_limit[0] < next_y < y_limit[1]):
                        new_actions[i] = 'rotl'
                        new_step_size[i] = 15
            except:
                pass
        return new_actions, new_step_size