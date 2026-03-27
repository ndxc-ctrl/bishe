import os
import sys
import json
import math
import copy
import random
import time
import numpy as np
import tqdm
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from src.common.param import args
from uav_utils.logger import logger

# 导入 Isaac Sim 核心 API
from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage, clear_stage
from omni.isaac.core.prims import XFormPrim
from omni.isaac.sensor import Camera
import omni.replicator.core as rep


# =========================================================================
# 常量和状态类
# =========================================================================
class AirsimActionSettings:
    FORWARD_STEP_SIZE = 500.0  # 每次飞 5 米
    LEFT_RIGHT_STEP_SIZE = 500.0
    UP_DOWN_STEP_SIZE = 200.0
    TURN_ANGLE = 15.0


class SimState:
    def __init__(self, index=-1, step=0, task_info=None):
        if task_info is None:
            task_info = {}
        self.index = index
        self.step = step
        self.task_info = task_info
        self.is_end = False
        self.oracle_success = False
        self.is_collisioned = False
        self.SUCCESS_DISTANCE = 20

        self.target_position = task_info.get('object_position', [[0, 0, 0]])[-1]
        self.start_pose = task_info.get('start_pose', {"start_position": [0, 0, 0], "start_quaternionr": [0, 0, 0, 1]})

        self.trajectory = [{'sensors': {
            'state': {
                'position': self.start_pose['start_position'],
                'quaternionr': self.start_pose['start_quaternionr']
            },
        },
            'move_distance': float(0.0),
            'distance_to_target': self.task_info.get('distance_to_target', 0.0)
        }]
        self.move_distance = 0.0
        self.heading_changes = []

    @property
    def state(self):
        return self.trajectory[-1]['sensors']['state']

    @property
    def pose(self):
        # 返回拼接的 list: [x, y, z, x, y, z, w]
        pos = self.trajectory[-1]['sensors']['state']['position']
        quat = self.trajectory[-1]['sensors']['state']['quaternionr']
        return pos + quat


def getNextPosition(current_position, current_orientation, action, step_size, is_fixed):
    current_position = np.array(current_position)
    r = R.from_quat(current_orientation)
    euler = r.as_euler('xyz', degrees=False)
    roll, pitch, yaw = euler[0], euler[1], euler[2]

    if action == 'forward':
        dx, dy, dz = math.cos(yaw), math.sin(yaw), 0
        vector = np.array([dx, dy, dz])
        norm = np.linalg.norm(vector)
        unit_vector = vector / norm if norm > 1e-6 else np.array([0, 0, 0])
        step = 500.0  # 强制写死 500
        new_position = current_position + unit_vector * step
        new_orientation = current_orientation
        fly_type = "move"

    elif action == 'rotl':
        step = 15.0  # 强制写死 15度
        new_yaw = yaw - math.radians(step)
        new_position = current_position
        new_orientation = R.from_euler('xyz', [roll, pitch, new_yaw]).as_quat()
        fly_type = "rotate"

    elif action == 'rotr':
        step = 15.0
        new_yaw = yaw + math.radians(step)
        new_position = current_position
        new_orientation = R.from_euler('xyz', [roll, pitch, new_yaw]).as_quat()
        fly_type = "rotate"

    elif action == 'ascend':
        unit_vector = np.array([0, 0, 1])  # Isaac Sim 默认 Z 轴朝上
        step = 200.0
        new_position = current_position + unit_vector * step
        new_orientation = current_orientation
        fly_type = "move"

    elif action == 'descend':
        unit_vector = np.array([0, 0, -1])
        step = 200.0
        new_position = current_position + unit_vector * step
        new_orientation = current_orientation
        fly_type = "move"

    elif action == 'left':
        unit_x = 1.0 * math.cos(yaw + math.pi / 2)
        unit_y = 1.0 * math.sin(yaw + math.pi / 2)
        vector = np.array([unit_x, unit_y, 0])
        norm = np.linalg.norm(vector)
        unit_vector = vector / norm if norm > 1e-6 else np.array([0, 0, 0])
        step = 500.0
        new_position = current_position - unit_vector * step
        new_orientation = current_orientation
        fly_type = "move"

    elif action == 'right':
        unit_x = 1.0 * math.cos(yaw + math.pi / 2)
        unit_y = 1.0 * math.sin(yaw + math.pi / 2)
        vector = np.array([unit_x, unit_y, 0])
        norm = np.linalg.norm(vector)
        unit_vector = vector / norm if norm > 1e-6 else np.array([0, 0, 0])
        step = 500.0
        new_position = current_position + unit_vector * step
        new_orientation = current_orientation
        fly_type = "move"

    else:  # stop
        new_position = current_position
        new_orientation = current_orientation
        fly_type = "stop"

    return new_position, new_orientation, fly_type


# =========================================================================
# Isaac Sim 环境类
# =========================================================================
class AirVLNENV:
    def __init__(self, batch_size=8, dataset_path=None, save_path=None, seed=1, activate_maps=[]):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.epoch_done = False
        self.seed = seed
        self.activate_maps = set(activate_maps)
        self.exist_save_path = save_path

        # 初始化 Isaac Sim 世界
        self.world = World()
        self.drone_prims = []
        self.camera_rigs = []

        load_data = self.load_my_datasets()
        self.data = load_data
        logger.info('Loaded dataset {}.'.format(len(self.data)))
        self.index_data = 0
        self.data = self._group_scenes()
        logger.info('dataset grouped by scene.')

        scenes = [item['map_name'] for item in self.data]
        self.scenes = set(scenes)
        self.sim_states = [None for _ in range(batch_size)]
        self.last_using_map = None

    def load_my_datasets(self):
        data = []
        with open(self.dataset_path, 'r') as f:
            data_file = json.load(f)

        for index, item in enumerate(tqdm.tqdm(data_file, desc="Loading")):
            traj_info = {}
            traj_info['task_id'] = item.get('episode_id', f"task_{index}")
            traj_info['map_name'] = item.get('map_name', 'CityDemopack')
            traj_info['object_name'] = item.get('true_name', 'Anomaly')
            traj_info['object_position'] = item.get('pose', [[0, 0, 0]])
            traj_info['start_pose'] = item.get('start_pose', {
                "start_position": [0, 0, 0],
                "start_quaternionr": [0, 0, 0, 1]
            })
            traj_info['description'] = item.get('description', '')
            info_field = item.get('info', {})
            traj_info['distance_to_target'] = info_field.get('euclidean_distance', 0.0)
            traj_info['trajectory_dir'] = self.dataset_path
            data.append(traj_info)
        return data

    def _group_scenes(self):
        return sorted(self.data, key=lambda e: str(e['map_name']))

    def next_minibatch(self, skip_scenes=[], data_it=0):
        batch = []
        if self.epoch_done and self.index_data >= len(self.data):
            self.batch = None
            return

        while True:
            if self.index_data >= len(self.data):
                self.epoch_done = True
                random.shuffle(self.data)
                self.data = self._group_scenes()
                if len(batch) == 0:
                    self.index_data = 0
                    self.batch = None
                    return
                self.index_data = self.batch_size - len(batch)
                batch += self.data[:self.index_data]
                self.index_data = len(self.data) + 1
                break

            task = self.data[self.index_data]
            if task['map_name'] in skip_scenes:
                self.index_data += 1
                continue

            batch.append(task)
            self.index_data += 1

            if len(batch) == self.batch_size:
                break

        self.batch = copy.deepcopy(batch)
        return self.batch

    def changeToNewTask(self):
        self._changeEnv()
        self._setDrone()
        self.update_measurements()

    def _changeEnv(self):
        current_map = self.batch[0]['map_name']
        if current_map == self.last_using_map:
            return

        logger.warning(f'Changing env to: {current_map}')
        self.world.clear_instance()
        clear_stage()

        usd_path = os.path.abspath(f"./TEST_ENVS/World_{current_map}.usd")
        if not os.path.exists(usd_path):
            usd_path = os.path.abspath(f"./TRAIN_ENVS/World_{current_map}.usd")

        open_stage(usd_path)
        self.world = World()

        self.drone_prims = []
        self.camera_rigs = []

        for i in range(self.batch_size):
            drone_path = f"/World/Drone_{i}"
            drone = XFormPrim(prim_path=drone_path, name=f"drone_{i}")
            self.drone_prims.append(drone)

            cams = []
            cam_configs = [
                ("Front", [0, 0, 0]),
                ("Left", [0, 0, 90]),
                ("Right", [0, 0, -90]),
                ("Down", [0, -90, 0])
            ]
            for name, rot_euler in cam_configs:
                cam_path = f"{drone_path}/Camera_{name}"
                quat = R.from_euler('xyz', rot_euler, degrees=True).as_quat()
                cam = Camera(
                    prim_path=cam_path,
                    translation=np.array([0, 0, 0]),
                    orientation=np.array([quat[3], quat[0], quat[1], quat[2]]),
                    resolution=(640, 480)
                )
                cam.initialize()
                cam.add_distance_to_image_plane_to_frame()
                cams.append(cam)
            self.camera_rigs.append(cams)

        self.world.reset()
        self.last_using_map = current_map

    def _setDrone(self):
        for i, item in enumerate(self.batch):
            pos = item['start_pose']['start_position']
            target = item.get('object_position', [[0, 0, 0]])[-1]

            # --- 自动瞄准逻辑开始 ---
            dx = target[0] - pos[0]
            dy = target[1] - pos[1]
            correct_yaw = math.atan2(dy, dx)
            r = R.from_euler('xyz', [0, 0, correct_yaw], degrees=False)
            quat = r.as_quat()
            # --- 自动瞄准逻辑结束 ---

            isaac_quat = np.array([quat[3], quat[0], quat[1], quat[2]])
            self.drone_prims[i].set_world_pose(position=np.array(pos), orientation=isaac_quat)
            self.sim_states[i] = SimState(index=i, step=0, task_info=item)

        self.world.step(render=True)

    def get_obs(self):
        states = self._getStates()
        outputs = []
        new_sim_states = []
        for i, state_tuple in enumerate(states):
            rgb_images, depth_images, sim_state = state_tuple

            pre_poses = []
            for t in sim_state.trajectory:
                pre_poses.append({
                    'position': t['sensors']['state']['position'],
                    'quaternionr': t['sensors']['state']['quaternionr'],
                    'sensors': t['sensors']
                })

                desc = sim_state.task_info.get('description', '')
                if not desc:
                    desc = sim_state.task_info.get('instruction', '')

                obs_dict = {
                    'images': rgb_images,
                    'rgb': rgb_images,
                    'depth': depth_images,
                    'pose': sim_state.pose,
                    'sensors': {
                        'state': {
                            'position': sim_state.pose[:3],
                            'quaternionr': sim_state.pose[3:]
                        }
                    },
                    'step': sim_state.step,
                    'description': desc,
                    'instruction': desc,
                    'object_name': sim_state.task_info.get('object_name', ''),
                    'object_size': sim_state.task_info.get('object_size', ''),
                    'pre_poses': pre_poses,
                    'move_distance': float(sim_state.move_distance),
                    'avg_heading_changes': float(
                        np.mean(sim_state.heading_changes)) if sim_state.heading_changes else 0.0,
                    'start_position': sim_state.start_pose.get('start_position', [0, 0, 0]),
                    'start_quaternionr': sim_state.start_pose.get('start_quaternionr', [0, 0, 0, 1]),
                }

            outputs.append((
                [obs_dict],
                sim_state.is_end,
                sim_state.is_collisioned,
                sim_state.oracle_success
            ))
            new_sim_states.append(sim_state)

        self.sim_states = new_sim_states
        return outputs

    def _getStates(self):
        self.world.step(render=True)
        states = []
        for i in range(self.batch_size):
            rgb_images = []
            depth_images = []
            for cam in self.camera_rigs[i]:
                rgba = cam.get_rgba()

                # 【终极修复】：极其小心地处理来自底层的图像数据
                if rgba is not None:
                    # 强制转成独立的 numpy 数组，防止 C++ 底层内存被意外释放导致段错误(核心已转储)
                    rgba_np = np.array(rgba)

                    # 如果变成 1D 数组，先尝试安全重塑
                    if getattr(rgba_np, 'ndim', 1) == 1:
                        # 确保长度是 480*640*4
                        if len(rgba_np) == 480 * 640 * 4:
                            rgba_np = rgba_np.reshape((480, 640, 4))
                        else:
                            # 长度不对，强制变黑图
                            rgba_np = np.zeros((480, 640, 4), dtype=np.uint8)

                    # 再次安全校验，如果真的是 3D 的，才去切片
                    if getattr(rgba_np, 'ndim', 1) == 3 and rgba_np.shape[2] >= 3:
                        rgb_images.append(np.array(rgba_np[:, :, :3], dtype=np.uint8))
                    else:
                        rgb_images.append(np.zeros((480, 640, 3), dtype=np.uint8))
                else:
                    rgb_images.append(np.zeros((480, 640, 3), dtype=np.uint8))

                # 获取深度图
                depth = cam.get_current_frame().get("distance_to_image_plane")
                if depth is not None:
                    depth_normalized = (np.clip(depth, 0, 100) / 100 * 255).astype(np.uint8)
                    depth_images.append(depth_normalized)
                else:
                    depth_images.append(np.zeros((480, 640), dtype=np.uint8))

            states.append((rgb_images, depth_images, self.sim_states[i]))
        return states

    def reset(self):
        self.changeToNewTask()
        return self.get_obs()

    def makeActions(self, action_list, steps_size, is_fixed):
        env_dones = []
        rewards = []

        for i, action in enumerate(action_list):
            if self.sim_states[i].is_end:
                env_dones.append(True)
                rewards.append(0)
                continue

            if action == 'stop' or self.sim_states[i].step >= int(args.maxActions):
                self.sim_states[i].is_end = True

            current_pos = self.sim_states[i].pose[:3]
            current_quat = self.sim_states[i].pose[3:]

            new_pos, new_quat, fly_type = getNextPosition(
                current_pos, current_quat, action, steps_size[i], is_fixed
            )

            # 更新偏航角变化量 (Heading Changes)
            r_old = R.from_quat(current_quat)
            r_new = R.from_quat(new_quat)
            yaw_old = r_old.as_euler('xyz', degrees=True)[2]
            yaw_new = r_new.as_euler('xyz', degrees=True)[2]
            delta_yaw = abs((yaw_new - yaw_old + 180) % 360 - 180)
            self.sim_states[i].heading_changes.append(delta_yaw)

            isaac_quat = np.array([new_quat[3], new_quat[0], new_quat[1], new_quat[2]])
            self.drone_prims[i].set_world_pose(position=np.array(new_pos), orientation=isaac_quat)

            self.sim_states[i].step += 1
            self.sim_states[i].move_distance += np.linalg.norm(np.array(new_pos) - np.array(current_pos))

            target = np.array(self.sim_states[i].target_position)
            distance_to_target = float(np.linalg.norm(np.array(new_pos) - target))

            if distance_to_target < self.sim_states[i].SUCCESS_DISTANCE:
                self.sim_states[i].oracle_success = True

            self.sim_states[i].trajectory.append({
                'sensors': {
                    'state': {
                        'position': list(new_pos),
                        'quaternionr': list(new_quat)
                    }
                },
                'move_distance': round(self.sim_states[i].move_distance, 2),
                'distance_to_target': round(distance_to_target, 2),
            })

            self.sim_states[i].is_collisioned = False

            env_dones.append(self.sim_states[i].is_end)
            rewards.append(0)

        self.world.step(render=True)
        new_obs = self.get_obs()
        return new_obs, rewards, env_dones, {}

    def update_measurements(self):
        for idx in range(len(self.batch)):
            curr = np.array(self.sim_states[idx].pose[:3])
            coords = np.array(self.batch[idx]['object_position'])
            dist = np.linalg.norm(coords - curr[None, :], axis=1).min() if coords.ndim == 2 else np.linalg.norm(
                curr - coords)
            print(
                f'batch[{idx}/{len(self.batch)}]| distance: {round(float(dist), 2)}, position: {curr[0]}, {curr[1]}, {curr[2]}, target: {coords}')

    def delete_VectorEnvUtil(self):
        pass

    def close(self):
        self.world.clear_instance()