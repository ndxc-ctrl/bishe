from collections import OrderedDict
import copy
import random
import sys
import time
import numpy as np
import math
import os
import json
import gc
from pathlib import Path
from typing import Dict, List, Optional
import tqdm

# 恢复使用前缀！正常导入！
from src.common.param import args
from uav_utils.logger import logger

sys.path.append(str(Path(str(os.getcwd())).resolve()))
from airsim_plugin.AirVLNSimulatorClientTool import AirVLNSimulatorClientTool
from uav_utils.env_utils_uav import SimState, getNextPosition
from uav_utils.env_vector_uav import VectorEnvUtil

# =========================================================================
# 彻底摆脱 airsim 依赖！自己实现简单的数学坐标类，专供 Isaac Sim 传递数据
# =========================================================================
class Vector3r:
    def __init__(self, x_val=0.0, y_val=0.0, z_val=0.0):
        self.x_val = x_val
        self.y_val = y_val
        self.z_val = z_val

class Quaternionr:
    def __init__(self, x_val=0.0, y_val=0.0, z_val=0.0, w_val=1.0):
        self.x_val = x_val
        self.y_val = y_val
        self.z_val = z_val
        self.w_val = w_val

class Pose:
    def __init__(self, position_val=None, orientation_val=None):
        self.position = position_val if position_val else Vector3r()
        self.orientation = orientation_val if orientation_val else Quaternionr()

class KinematicsState:
    def __init__(self):
        self.position = Vector3r()
        self.orientation = Quaternionr()
        self.linear_velocity = Vector3r()
        self.angular_velocity = Vector3r()

def to_eularian_angles(q):
    sinr_cosp = 2 * (q.w_val * q.x_val + q.y_val * q.z_val)
    cosr_cosp = 1 - 2 * (q.x_val * q.x_val + q.y_val * q.y_val)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (q.w_val * q.y_val - q.z_val * q.x_val)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy_cosp = 1 - 2 * (q.y_val * q.y_val + q.z_val * q.z_val)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return pitch, roll, yaw

# =========================================================================

class AirVLNENV:
    def __init__(self, batch_size=8, dataset_path=None, save_path=None, seed=1, activate_maps=[]):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.epoch_done = False
        self.seed = seed
        self.collected_keys = set()
        self.activate_maps = set(activate_maps)
        self.exist_save_path = save_path
        load_data = self.load_my_datasets()
        self.data = load_data
        logger.info('Loaded dataset {}.'.format(len(self.data)))
        self.index_data = 0
        self.dataset_group_by_scene = True
        self.data = self._group_scenes()
        logger.info('dataset grouped by scene, ')

        scenes = [item['map_name'] for item in self.data]
        self.scenes = set(scenes)
        self.sim_states: Optional[List[SimState]] = [None for _ in range(batch_size)]
        self.last_using_map_list = []
        self.one_scene_could_use_num = 5e3
        self.this_scene_used_cnt = 0
        self.init_VectorEnvUtil()

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
        scene_sort_keys: OrderedDict[str, int] = {}
        for item in self.data:
            if str(item['map_name']) not in scene_sort_keys:
                scene_sort_keys[str(item['map_name'])] = len(scene_sort_keys)
        return sorted(self.data, key=lambda e: (scene_sort_keys[str(e['map_name'])]))

    def init_VectorEnvUtil(self):
        self.delete_VectorEnvUtil()
        self.VectorEnvUtil = VectorEnvUtil(self.scenes, self.batch_size)

    def delete_VectorEnvUtil(self):
        if hasattr(self, 'VectorEnvUtil'):
            del self.VectorEnvUtil
        import gc
        gc.collect()

    def next_minibatch(self, skip_scenes=[], data_it=0):
        batch = []
        if self.epoch_done and self.index_data >= len(self.data):
            self.batch = None
            return

        while True:
            if self.index_data >= len(self.data):
                self.epoch_done = True
                random.shuffle(self.data)
                logger.warning('random shuffle data and pad to batch size')
                if self.dataset_group_by_scene:
                    self.data = self._group_scenes()
                    logger.warning('dataset grouped by scene')

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
        assert len(self.batch) == self.batch_size, 'next_minibatch error'
        self.VectorEnvUtil.set_batch(self.batch)
        return self.batch

    def changeToNewTask(self):
        self._changeEnv(need_change=False)
        self._setDrone()
        self.update_measurements()

    def _setDrone(self, ):
        drone_position_info = [item['start_pose']['start_position'] for item in self.batch]
        drone_quaternior_info = [item['start_pose']['start_quaternionr'] for item in self.batch]
        poses = []
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            poses.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                pose = Pose(
                    position_val=Vector3r(
                        x_val=drone_position_info[cnt][0],
                        y_val=drone_position_info[cnt][1],
                        z_val=drone_position_info[cnt][2],
                    ),
                    orientation_val=Quaternionr(
                        x_val=drone_quaternior_info[cnt][0],
                        y_val=drone_quaternior_info[cnt][1],
                        z_val=drone_quaternior_info[cnt][2],
                        w_val=drone_quaternior_info[cnt][3],
                    ),
                )
                poses[index_1].append(pose)
                cnt += 1

        self.simulator_tool.setPoses(poses=poses)
        state_info_results = self.simulator_tool.getSensorInfo()

        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            for index_2, _ in enumerate(item['open_scenes']):
                self.sim_states[cnt] = SimState(index=cnt, step=0, task_info=self.batch[cnt])
                self.sim_states[cnt].sensorInfo = [state_info_results[index_1][index_2]]
                cnt += 1

    def _changeEnv(self, need_change: bool = True):
        using_map_list = [item['map_name'] for item in self.batch]
        assert len(using_map_list) == self.batch_size, '错误'

        machines_info_template = copy.deepcopy(args.machines_info)
        total_max_scene_num = 0
        for item in machines_info_template:
            total_max_scene_num += item['MAX_SCENE_NUM']
        assert self.batch_size <= total_max_scene_num, 'error args param: batch_size'

        machines_info = []
        ix = 0
        for index, item in enumerate(machines_info_template):
            machines_info.append(item)
            delta = min(self.batch_size, item['MAX_SCENE_NUM'], len(using_map_list) - ix)
            machines_info[index]['open_scenes'] = using_map_list[ix: ix + delta]

            machines_info[index]['gpus'] = [args.gpu_id] * len(machines_info[index]['open_scenes'])
            ix += delta

        cnt = 0
        for item in machines_info:
            cnt += len(item['open_scenes'])
        assert self.batch_size == cnt, 'error create machines_info'

        if self.this_scene_used_cnt < self.one_scene_could_use_num and \
                len(set(using_map_list)) == 1 and len(set(self.last_using_map_list)) == 1 and \
                using_map_list[0] is not None and self.last_using_map_list[0] is not None and \
                using_map_list[0] == self.last_using_map_list[0] and \
                need_change == False:
            self.this_scene_used_cnt += 1
            logger.warning('no need to change env: {}'.format(using_map_list))
            return
        else:
            logger.warning('to change env: {}'.format(using_map_list))

        while True:
            try:
                self.machines_info = copy.deepcopy(machines_info)
                print('machines_info:', self.machines_info)
                self.simulator_tool = AirVLNSimulatorClientTool(machines_info=self.machines_info)
                self.simulator_tool.run_call()
                break
            except Exception as e:
                logger.error("启动场景失败 {}".format(e))
                time.sleep(3)
            except:
                logger.error('启动场景失败')
                time.sleep(3)

        self.last_using_map_list = using_map_list.copy()
        self.this_scene_used_cnt = 1

    def get_obs(self):
        obs_states = self._getStates()
        obs, states = self.VectorEnvUtil.get_obs(obs_states)
        self.sim_states = states
        return obs

    def _getStates(self):
        responses = self.simulator_tool.getImageResponses()
        cnt = 0
        for item in responses:
            cnt += len(item)
        assert len(responses) == len(self.machines_info), 'error'
        assert cnt == self.batch_size, 'error'

        states = [None for _ in range(self.batch_size)]
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            for index_2 in range(len(item['open_scenes'])):
                rgb_images = responses[index_1][index_2][0]
                depth_images = responses[index_1][index_2][1]
                state = self.sim_states[cnt]
                states[cnt] = (rgb_images, depth_images, state)
                cnt += 1
        return states

    def _get_current_state(self) -> list:
        states = []
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            states.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                s = self.sim_states[cnt].state
                state = KinematicsState()
                state.position = Vector3r(*s['position'])
                state.orientation = Quaternionr(*s['orientation'])
                state.linear_velocity = Vector3r(*s['linear_velocity'])
                state.angular_velocity = Vector3r(*s['angular_velocity'])
                states[index_1].append(state)
                cnt += 1
        return states

    def _get_current_pose(self) -> list:
        poses = []
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            poses.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                poses[index_1].append(
                    self.sim_states[cnt].pose
                )
                cnt += 1
        return poses

    def reset(self):
        self.changeToNewTask()
        return self.get_obs()

    def makeActions(self, action_list, steps_size, is_fixed):
        poses = []
        fly_types = []
        for index, action in enumerate(action_list):
            if self.sim_states[index].is_end == True:
                action = 'stop'
            if action == 'stop' or self.sim_states[index].step >= int(args.maxActions):
                self.sim_states[index].is_end = True

            current_pose = self.sim_states[index].pose

            if isinstance(current_pose, list):
                pos = current_pose[:3]
                quat = current_pose[3:]
                airsim_pose = Pose(Vector3r(*pos),
                                   Quaternionr(x_val=quat[0], y_val=quat[1], z_val=quat[2],
                                               w_val=quat[3]))
            else:
                airsim_pose = current_pose

            (new_pose, fly_type) = getNextPosition(airsim_pose, action, steps_size[index], is_fixed)

            (prev_pitch, prev_roll, prev_yaw) = to_eularian_angles(airsim_pose.orientation)
            (curr_pitch, curr_roll, curr_yaw) = to_eularian_angles(new_pose.orientation)
            delta_yaw = abs((math.degrees(curr_yaw - prev_yaw) + 180) % 360 - 180)
            self.sim_states[index].heading_changes.append(delta_yaw)
            pos = new_pose.position

            curr = np.array([pos.x_val, pos.y_val, pos.z_val])
            coords = np.array(self.batch[index]["object_position"])
            if coords.ndim == 2 and coords.shape[1] == 3:
                dists = np.linalg.norm(coords - curr[None, :], axis=1)
                min_dist = dists.min()
            else:
                min_dist = np.linalg.norm(curr - coords)
            if min_dist < self.sim_states[index].SUCCESS_DISTANCE:
                self.sim_states[index].oracle_success = True

            poses.append(new_pose)
            fly_types.append(fly_type)

        format_pose = []
        format_fly_type = []
        cnt = 0
        for index1, item in enumerate(self.machines_info):
            format_pose.append([])
            format_fly_type.append([])
            for index2, _ in enumerate(item['open_scenes']):
                format_pose[index1].append(poses[cnt])
                format_fly_type[index1].append(fly_types[cnt])
                cnt += 1

        result = self.simulator_tool.move_to_next_pose(poses_list=format_pose, fly_types=format_fly_type)

        if not result:
            logger.error('move_to_next_pose error')

        cnt = 0
        for index1, item in enumerate(self.machines_info):
            for index2, _ in enumerate(item['open_scenes']):
                self.sim_states[cnt].is_collisioned = result[index1][index2]['collision']
                cnt += 1

        for index, action in enumerate(action_list):
            if self.sim_states[index].is_end == True:
                continue

            if action == 'stop' or self.sim_states[index].step >= int(args.maxActions):
                self.sim_states[index].is_end = True

            self.sim_states[index].step += 1

            traj = self.sim_states[index].trajectory
            if len(traj) >= 1:
                p_prev = np.array(traj[-1]['sensors']['state']['position'])
            else:
                p_prev = np.array([poses[index].position.x_val,
                                   poses[index].position.y_val,
                                   poses[index].position.z_val])
            p_curr = np.array([poses[index].position.x_val,
                               poses[index].position.y_val,
                               poses[index].position.z_val])
            self.sim_states[index].move_distance += np.linalg.norm(p_curr - p_prev)

            target = np.array(self.sim_states[index].target_position)
            distance_to_target = float(np.linalg.norm(p_curr - target))

            self.sim_states[index].trajectory.append({
                'sensors': {
                    'state': {
                        'position': [poses[index].position.x_val, poses[index].position.y_val,
                                     poses[index].position.z_val],
                        'quaternionr': [poses[index].orientation.x_val, poses[index].orientation.y_val,
                                        poses[index].orientation.z_val, poses[index].orientation.w_val]
                    }
                },
                'move_distance': round(self.sim_states[index].move_distance, 2),
                'distance_to_target': round(distance_to_target, 2),
            })

    def update_measurements(self):
        for idx in range(len(self.batch)):
            curr = np.array(self.sim_states[idx].pose[0:3] if isinstance(self.sim_states[idx].pose, list) else [self.sim_states[idx].pose.position.x_val, self.sim_states[idx].pose.position.y_val, self.sim_states[idx].pose.position.z_val])
            coords = np.array(self.batch[idx]['object_position'])
            dist = np.linalg.norm(coords - curr[None, :], axis=1).min() if coords.ndim == 2 else np.linalg.norm(curr - coords)
            print(f'batch[{idx}/{len(self.batch)}]| distance: {round(float(dist), 2)}, position: {curr[0]}, {curr[1]}, {curr[2]}, target: {coords}')