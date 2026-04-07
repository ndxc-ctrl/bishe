import os
import json
import time
import math
import torch
import numpy as np
import airsim
import torch.nn.functional as F

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

        proxy_url = "http://" + "127.0.0.1:13849"
        custom_http_client = httpx.Client(proxy=proxy_url)

        self.gpt_client = OpenAI(
            api_key="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",
            http_client=custom_http_client,
            timeout=30.0
        )

        self.start_position = [[] for _ in range(batch_size)]
        self.start_yaw = [0 for _ in range(batch_size)]
        self.current_poses = [[] for _ in range(batch_size)]

        self.waypoints = [[] for _ in range(batch_size)]
        self.current_wp_idx = [0 for _ in range(batch_size)]

    def generate_waypoints(self, poly):
        """
        根据给定的凸多边形坐标，在内部生成蛇形覆盖（Lawnmower）扫描航点
        """
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        min_y, max_y = min(ys), max(ys)

        # 【修改点】：将间距改为 5000.0，即 50 米侧移
        sweep_width = 5000.0
        waypoints = []
        y = min_y + sweep_width / 2
        direction = 1

        while y <= max_y:
            ints = []
            for i in range(len(poly)):
                p1 = poly[i]
                p2 = poly[(i + 1) % len(poly)]
                if min(p1[1], p2[1]) <= y <= max(p1[1], p2[1]):
                    if p1[1] != p2[1]:
                        x_int = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                        ints.append(x_int)
            ints.sort()

            if len(ints) >= 2:
                x_start, x_end = ints[0], ints[-1]
                x_start += 200
                x_end -= 200
                if x_start < x_end:
                    if direction == 1:
                        waypoints.extend([(x_start, y), (x_end, y)])
                    else:
                        waypoints.extend([(x_end, y), (x_start, y)])
                    direction *= -1
            y += sweep_width
        return waypoints

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

            search_area = episodes[i][-1].get('search_area', None)
            if not self.waypoints[i] and search_area and len(search_area) == 4 and isinstance(search_area[0], list):
                self.waypoints[i] = self.generate_waypoints(search_area)
                self.current_wp_idx[i] = 0
                print(
                    f"🗺️ [底层导航]: 成功解析自定义多边形区域，已生成 {len(self.waypoints[i])} 个全覆盖扫描航点 (侧移5000单位)。")

            step_num = episodes[i][-1].get('step', 0)
            move_distance = episodes[i][-1].get('move_distance', 0.0)
            task_desc = episodes[i][-1].get('description', '')

            prompt_template = fixed_user_prompt_template if fixed else unfixed_user_prompt_template
            system_prompt = fixed_system_prompt if fixed else unfixed_system_prompt

            user_content = prompt_template.format(
                description=task_desc,
                captions4=captions4, depth_info=depth_info4,
                step_num=step_num, move_distance=move_distance
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
            if action == 'stop':
                new_actions[i] = 'stop'
                new_step_size[i] = 0.0
                continue

            curr_pos = self.current_poses[i]
            curr_x, curr_y, _, curr_yaw_deg = curr_pos
            wps = self.waypoints[i]

            while self.current_wp_idx[i] < len(wps):
                target_x, target_y = wps[self.current_wp_idx[i]]
                dist = math.hypot(target_x - curr_x, target_y - curr_y)
                if dist < 100.0:
                    self.current_wp_idx[i] += 1
                    print(f"✅ [底层导航]: 抵达航点 {self.current_wp_idx[i]}/{len(wps)}，侧移后准备反向扫描。")
                else:
                    break

            if self.current_wp_idx[i] >= len(wps):
                new_actions[i] = 'stop'
                new_step_size[i] = 0.0
                continue

            target_x, target_y = wps[self.current_wp_idx[i]]
            dx = target_x - curr_x
            dy = target_y - curr_y
            dist = math.hypot(dx, dy)

            target_yaw_rad = math.atan2(dy, dx)
            target_yaw_deg = math.degrees(target_yaw_rad)
            yaw_diff = (target_yaw_deg - curr_yaw_deg + 180) % 360 - 180

            if abs(yaw_diff) < 5.0:
                new_actions[i] = 'forward'
                new_step_size[i] = min(500.0, dist)
            else:
                new_actions[i] = 'rotl' if yaw_diff < 0 else 'rotr'
                turn_step = abs(yaw_diff)
                if turn_step > 45.0: turn_step = 45.0
                new_step_size[i] = round(turn_step, 1)

        return new_actions, new_step_size