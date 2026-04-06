import os
import sys
import json
import math
import copy
import random
import time
import numpy as np
import tqdm
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from src.common.param import args
from uav_utils.logger import logger

from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage, clear_stage, add_reference_to_stage, get_current_stage
from omni.isaac.core.prims import XFormPrim
from omni.isaac.sensor import Camera
import omni.replicator.core as rep
from pxr import Usd, UsdPhysics


class AirsimActionSettings:
    FORWARD_STEP_SIZE = 500.0
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
        step = step_size
        new_position = current_position + unit_vector * step
        new_orientation = current_orientation
        fly_type = "move"

    elif action == 'rotl':
        step = step_size
        new_yaw = yaw - math.radians(step)
        new_position = current_position
        new_orientation = R.from_euler('xyz', [roll, pitch, new_yaw]).as_quat()
        fly_type = "rotate"

    elif action == 'rotr':
        step = step_size
        new_yaw = yaw + math.radians(step)
        new_position = current_position
        new_orientation = R.from_euler('xyz', [roll, pitch, new_yaw]).as_quat()
        fly_type = "rotate"

    elif action == 'ascend':
        unit_vector = np.array([0, 0, 1])
        step = step_size
        new_position = current_position + unit_vector * step
        new_orientation = current_orientation
        fly_type = "move"

    elif action == 'descend':
        unit_vector = np.array([0, 0, -1])
        step = step_size
        new_position = current_position + unit_vector * step
        new_orientation = current_orientation
        fly_type = "move"

    elif action == 'left':
        unit_x = 1.0 * math.cos(yaw + math.pi / 2)
        unit_y = 1.0 * math.sin(yaw + math.pi / 2)
        vector = np.array([unit_x, unit_y, 0])
        norm = np.linalg.norm(vector)
        unit_vector = vector / norm if norm > 1e-6 else np.array([0, 0, 0])
        step = step_size
        new_position = current_position - unit_vector * step
        new_orientation = current_orientation
        fly_type = "move"

    elif action == 'right':
        unit_x = 1.0 * math.cos(yaw - math.pi / 2)
        unit_y = 1.0 * math.sin(yaw - math.pi / 2)
        vector = np.array([unit_x, unit_y, 0])
        norm = np.linalg.norm(vector)
        unit_vector = vector / norm if norm > 1e-6 else np.array([0, 0, 0])
        step = step_size
        new_position = current_position + unit_vector * step
        new_orientation = current_orientation
        fly_type = "move"

    else:
        new_position = current_position
        new_orientation = current_orientation
        fly_type = "stop"

    return new_position, new_orientation, fly_type


class AirVLNENV:
    def __init__(self, batch_size=8, dataset_path=None, save_path=None, seed=1, activate_maps=[]):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.epoch_done = False
        self.seed = seed
        self.activate_maps = set(activate_maps)
        self.exist_save_path = save_path

        self.world = World(stage_units_in_meters=1.0)
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
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            return [{
                'episode_id': 'test_0',
                'map_name': 'TestMap',
                'true_name': 'TestTarget',
                'pose': [[0, 0, 0]],
                'start_pose': {"start_position": [0, 0, 50], "start_quaternionr": [0, 0, 0, 1]},
                'description': 'Test env',
                'search_area': [[0, 0], [10000, 0], [10000, 10000], [0, 10000]]
            }]

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
            traj_info['search_area'] = item.get('search_area', None)

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

        ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        usd_path = os.path.join(ROOT_DIR, "TEST_ENVS", f"World_{current_map}.usd")

        if not os.path.exists(usd_path):
            usd_path = os.path.join(ROOT_DIR, "TRAIN_ENVS", f"World_{current_map}.usd")

        if os.path.exists(usd_path):
            open_stage(usd_path)
        else:
            self.world = World(stage_units_in_meters=1.0)
            self.world.scene.add_default_ground_plane()

        self.world = World()
        self.drone_prims = []
        self.camera_rigs = []

        stage = get_current_stage()
        iris_model_path = "/home/yczheng/UAV_ON-main/TEST_ENVS/iris.usd"

        for i in range(self.batch_size):
            drone_path = f"/World/Drone_{i}"
            drone_root = XFormPrim(prim_path=drone_path, name=f"drone_{i}")
            self.drone_prims.append(drone_root)

            vis_path = f"{drone_path}/visual_mesh"
            try:
                add_reference_to_stage(usd_path=iris_model_path, prim_path=vis_path)
                prim = stage.GetPrimAtPath(vis_path)

                for p in Usd.PrimRange(prim):
                    if p.HasAPI(UsdPhysics.ArticulationRootAPI): p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                    if p.HasAPI(UsdPhysics.RigidBodyAPI): p.RemoveAPI(UsdPhysics.RigidBodyAPI)
                    if p.HasAPI(UsdPhysics.CollisionAPI): p.RemoveAPI(UsdPhysics.CollisionAPI)
                    if p.IsA(UsdPhysics.Joint): p.SetActive(False)

                visual_prim = XFormPrim(prim_path=vis_path, name=f"drone_vis_{i}")
                visual_prim.set_local_scale(np.array([100.0, 100.0, 100.0]))
            except Exception as e:
                pass

            cams = []
            cam_configs = [
                ("Front", [0, 0, 0], [50, 0, 5]),
                ("Left", [0, 0, 90], [0, 50, 5]),
                ("Right", [0, 0, -90], [0, -50, 5]),
                ("Down", [0, 90, 0], [0, 0, -20]),
                ("ThirdPerson", [0, 15, 0], [-350, 0, 120])
            ]

            for name, rot_euler, trans in cam_configs:
                cam_path = f"{drone_path}/Camera_{name}"
                quat = R.from_euler('xyz', rot_euler, degrees=True).as_quat()
                cam = Camera(
                    prim_path=cam_path,
                    translation=np.array(trans),
                    orientation=np.array([quat[3], quat[0], quat[1], quat[2]]),
                    resolution=(640, 480)
                )
                cam.initialize()
                cam.add_distance_to_image_plane_to_frame()
                cams.append(cam)
            self.camera_rigs.append(cams)

        self.world.reset()
        import omni.timeline
        omni.timeline.get_timeline_interface().play()
        self.world.play()
        for _ in range(60):
            self.world.step(render=True)
        self.last_using_map = current_map

    def _setDrone(self):
        for i, item in enumerate(self.batch):
            pos = item['start_pose']['start_position']
            target = item.get('object_position', [[0, 0, 0]])[-1]

            try:
                target_np = np.array(target)
                debug_pos = target_np + np.array([0, 0, 4000.0])  # 调整为目标正上方 40 米

                isaac_debug_quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z 水平姿态

                self.drone_prims[i].set_world_pose(position=debug_pos, orientation=isaac_debug_quat)

                print(f"⏳ [火焰调试拍照]: 正在目标上方悬停，预热 4 秒等待 RTX Flow 体积粒子渲染...")

                import omni.kit.app
                import omni.timeline
                omni.timeline.get_timeline_interface().play()

                # 狂转 200 帧，强制刷新物理与系统 UI，这是压榨 Flow 特效出现的杀手锏
                for _ in range(200):
                    self.world.step(render=True)
                    omni.kit.app.get_app().update()

                cam_down = self.camera_rigs[i][3]
                rgba = cam_down.get_rgba()
                if rgba is not None:
                    img = np.array(rgba)
                    if len(img) == 480 * 640 * 4: img = img.reshape((480, 640, 4))
                    if img.ndim == 3 and img.shape[2] >= 3:
                        bgr = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
                        filename = f"debug_target_fire_task_{item.get('episode_id', i)}.png"
                        cv2.imwrite(filename, bgr)
                        print(f"🔥 [火焰调试拍照成功]: 目标上方俯视图已保存为 '{filename}'，这次应该能看到火了！")
            except Exception as e:
                print(f"⚠️ [火焰调试拍照失败]: {e}")

            dx = target[0] - pos[0]
            dy = target[1] - pos[1]
            correct_yaw = math.atan2(dy, dx)
            r = R.from_euler('xyz', [0, 0, correct_yaw], degrees=False)
            quat = r.as_quat()

            isaac_quat = np.array([quat[3], quat[0], quat[1], quat[2]])
            self.drone_prims[i].set_world_pose(position=np.array(pos), orientation=isaac_quat)
            self.sim_states[i] = SimState(index=i, step=0, task_info=item)

        for _ in range(10):
            self.world.step(render=True)

    def get_obs(self):
        states = self._getStates()
        outputs = []
        new_sim_states = []

        POOL_H, POOL_W = 120, 160

        for i, state_tuple in enumerate(states):
            rgb_images, depth_images, sim_state = state_tuple

            full_rgbd_list = []
            pooled_rgb_list = []
            pooled_depth_list = []

            for rgb, depth in zip(rgb_images[:4], depth_images[:4]):
                depth_expanded = np.expand_dims(depth, axis=-1)
                rgbd = np.concatenate((rgb, depth_expanded), axis=-1)
                full_rgbd_list.append(rgbd)

                pooled_rgb = cv2.resize(rgb, (POOL_W, POOL_H), interpolation=cv2.INTER_AREA)
                pooled_depth = cv2.resize(depth, (POOL_W, POOL_H), interpolation=cv2.INTER_AREA)

                pooled_rgb_list.append(pooled_rgb)
                pooled_depth_list.append(pooled_depth)

            pre_poses = []
            for t in sim_state.trajectory:
                pre_poses.append({
                    'position': t['sensors']['state']['position'],
                    'quaternionr': t['sensors']['state']['quaternionr'],
                    'sensors': t['sensors']
                })

            desc = sim_state.task_info.get('description', '')

            obs_dict = {
                'images': rgb_images[:4],
                'rgb': rgb_images[:4],
                'depth': depth_images[:4],
                'rgb_third_person': rgb_images[4] if len(rgb_images) > 4 else None,

                'full_rgbd': full_rgbd_list,
                'pooled_rgb': pooled_rgb_list,
                'pooled_depth': pooled_depth_list,

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
                'search_area': sim_state.task_info.get('search_area', None),
                'pre_poses': pre_poses,
                'move_distance': float(sim_state.move_distance),
                'avg_heading_changes': float(np.mean(sim_state.heading_changes)) if sim_state.heading_changes else 0.0,
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
                if rgba is not None:
                    rgba_np = np.array(rgba)
                    if getattr(rgba_np, 'ndim', 1) == 1:
                        if len(rgba_np) == 480 * 640 * 4:
                            rgba_np = rgba_np.reshape((480, 640, 4))
                        else:
                            rgba_np = np.zeros((480, 640, 4), dtype=np.uint8)

                    if getattr(rgba_np, 'ndim', 1) == 3 and rgba_np.shape[2] >= 3:
                        rgb_images.append(np.array(rgba_np[:, :, :3], dtype=np.uint8))
                    else:
                        rgb_images.append(np.zeros((480, 640, 3), dtype=np.uint8))
                else:
                    rgb_images.append(np.zeros((480, 640, 3), dtype=np.uint8))

                depth = cam.get_current_frame().get("distance_to_image_plane")
                if depth is not None:
                    MAX_DEPTH = 10000.0
                    depth[depth == np.inf] = MAX_DEPTH
                    depth_normalized = (np.clip(depth, 0, MAX_DEPTH) / MAX_DEPTH * 255).astype(np.uint8)
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

            if action == 'stop' or self.sim_states[i].step >= 2000:
                self.sim_states[i].is_end = True

            current_pos = self.sim_states[i].pose[:3]
            current_quat = self.sim_states[i].pose[3:]

            new_pos, new_quat, fly_type = getNextPosition(current_pos, current_quat, action, steps_size[i], is_fixed)

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
                'sensors': {'state': {'position': list(new_pos), 'quaternionr': list(new_quat)}},
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
        pass

    def delete_VectorEnvUtil(self):
        pass

    def close(self):
        self.world.clear_instance()