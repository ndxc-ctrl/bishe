from model_wrapper.base_model import BaseModelWrapper
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import io
import torch
import numpy as np
import math
from scipy.spatial.transform import Rotation as R


class CLIP_H(BaseModelWrapper):
    def __init__(self, batch_size):
        self.device = 'cuda:0'
        self.model, self.processor = self.load_clip()
        self.action_mapping = {
            0: 'forward',
            1: 'left',
            2: 'right',
            3: 'descend',
        }
        # 巡检任务的阈值可能需要根据实际情况调高或调低
        self.threshold = 0.24
        self.start_position = [[] for _ in range(batch_size)]
        self.start_yaw = [0 for _ in range(batch_size)]
        self.current_poses = [[] for _ in range(batch_size)]
        self.prev_action = [None for _ in range(batch_size)]

        self.DEFAULT_STEP = 5.0  # 替代 AirsimActionSettings

    def load_clip(self, ):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        return model.to(self.device), processor

    def prepare_inputs(self, episodes):
        images, depths, inputs, user_prompts = [], [], [], [[] for _ in range(len(episodes))]
        for i in range(len(episodes)):
            sources = episodes[i]
            for src in sources[::-1]:
                if 'rgb' in src and 'depth' in src:
                    images.extend(src['rgb'])
                    depths.append(src['depth'])
                    break

        for i in range(len(episodes)):
            self.start_position[i] = episodes[i][-1]['start_position']
            raw_poses = self.process_poses(poses=episodes[i][-1]['pre_poses'])

            if 0 < len(raw_poses) < 10:
                raw_poses += [raw_poses[-1]] * (10 - len(raw_poses))
            elif len(raw_poses) == 0:
                raw_poses = [[(self.start_position[i][0], self.start_position[i][1], self.start_position[i][2]),
                              self.start_yaw[i]]] * 10

            if len(raw_poses) > 0:
                self.current_poses[i] = [raw_poses[-1][0][0], raw_poses[-1][0][1], raw_poses[-1][0][2],
                                         raw_poses[-1][1]]
            else:
                self.current_poses[i] = [self.start_position[i][0], self.start_position[i][1],
                                         self.start_position[i][2], self.start_yaw[i]]

            pil_images = [Image.open(io.BytesIO(img)) for img in images[4 * i:4 * i + 4]]
            descriptions = episodes[i][-1]['description']
            proc_input = self.processor(text=descriptions, images=pil_images, return_tensors="pt", padding=True)
            inputs.append({k: v.to(self.device) for k, v in proc_input.items()})

        return inputs, user_prompts, depths

    def run(self, input, depth):
        actions, conflict, dones = [], [False] * len(input), []
        processed_depth = self.process_depth(depth)

        for i in range(len(input)):
            with torch.no_grad():
                outputs = self.model(**input[i])
                img_feats, txt_feats = outputs.image_embeds, outputs.text_embeds

                img_feats = img_feats / img_feats.norm(dim=1, keepdim=True)
                txt_feats = txt_feats / txt_feats.norm(dim=1, keepdim=True)
                cos_sim = img_feats @ txt_feats.T

                val, idx = cos_sim.topk(4, dim=0)
                action = self.action_mapping.get(int(idx[0].item()), 'forward')
                sim_value = val[0].item()

                if sim_value >= self.threshold:
                    action = 'stop'
                    print(f"🚨 警报: CLIP 判断画面高度匹配异常描述！(相似度: {sim_value:.2f})")

                prev_action = self.prev_action[i]
                if (prev_action == 'left' and action == 'right') or (prev_action == 'right' and action == 'left'):
                    action = self.action_mapping.get(int(idx[1].item()), 'forward')
                    conflict[i] = True

                if processed_depth[i] <= 5 and action == 'descend':
                    action = self.action_mapping.get(int(idx[2].item() if conflict[i] else idx[1].item()), 'forward')

                new_action = self.redirect_action(action, i)
                self.prev_action[i] = new_action
                actions.append(new_action)
                dones.append(action == 'stop')

        return actions, np.zeros(len(actions)) + self.DEFAULT_STEP, dones

    def process_depth(self, depth_images):
        return [np.min(np.array(depth) / 255.0 * 100.0).astype(int) for depth in depth_images]

    def process_poses(self, poses):
        pre_poses_xyzYaw = []
        for pose in poses:
            qx, qy, qz, qw = pose['quaternionr']
            yaw = R.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees=False)[2]
            pre_poses_xyzYaw.append(
                [(pose['position'][0], pose['position'][1], pose['position'][2]), round(math.degrees(yaw), 2)])
        return pre_poses_xyzYaw

    def redirect_action(self, action, i):
        new_action = action
        try:
            x_min, x_max = self.start_position[i][0] - 50, self.start_position[i][0] + 50
            y_min, y_max = self.start_position[i][1] - 50, self.start_position[i][1] + 50
            x, y, z, yaw = self.current_poses[i]

            if action in ['forward', 'left', 'right']:
                angle_offset = {'forward': 0, 'left': 90, 'right': -90}[action]
                rad = math.radians(yaw + angle_offset)
                unit_vector = np.array([math.cos(rad), math.sin(rad), 0])
                new_position = np.array([x, y, z]) + unit_vector * self.DEFAULT_STEP

                if new_position[0] > x_max or new_position[0] < x_min or new_position[1] > y_max or new_position[
                    1] < y_min:
                    new_action = 'rotl'
        except Exception:
            pass
        return new_action