import math
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from src.common.param import args
from uav_utils.logger import logger


# 固定步长常量
class AirsimActionSettings:
    FORWARD_STEP_SIZE = 5.0
    LEFT_RIGHT_STEP_SIZE = 5.0
    UP_DOWN_STEP_SIZE = 2.0
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
        step = AirsimActionSettings.FORWARD_STEP_SIZE if is_fixed else step_size
        new_position = current_position + unit_vector * step
        new_orientation = current_orientation
        fly_type = "move"

    elif action == 'rotl':
        step = AirsimActionSettings.TURN_ANGLE if is_fixed else step_size
        new_yaw = yaw - math.radians(step)
        new_position = current_position
        new_orientation = R.from_euler('xyz', [roll, pitch, new_yaw]).as_quat()
        fly_type = "rotate"

    elif action == 'rotr':
        step = AirsimActionSettings.TURN_ANGLE if is_fixed else step_size
        new_yaw = yaw + math.radians(step)
        new_position = current_position
        new_orientation = R.from_euler('xyz', [roll, pitch, new_yaw]).as_quat()
        fly_type = "rotate"

    elif action == 'ascend':
        unit_vector = np.array([0, 0, 1])  # Isaac Sim 默认 Z 轴朝上
        step = AirsimActionSettings.UP_DOWN_STEP_SIZE if is_fixed else step_size
        new_position = current_position + unit_vector * step
        new_orientation = current_orientation
        fly_type = "move"

    elif action == 'descend':
        unit_vector = np.array([0, 0, -1])
        step = AirsimActionSettings.UP_DOWN_STEP_SIZE if is_fixed else step_size
        new_position = current_position + unit_vector * step
        new_orientation = current_orientation
        fly_type = "move"

    elif action == 'left':
        unit_x = 1.0 * math.cos(yaw + math.pi / 2)
        unit_y = 1.0 * math.sin(yaw + math.pi / 2)
        vector = np.array([unit_x, unit_y, 0])
        norm = np.linalg.norm(vector)
        unit_vector = vector / norm if norm > 1e-6 else np.array([0, 0, 0])
        step = AirsimActionSettings.LEFT_RIGHT_STEP_SIZE if is_fixed else step_size
        new_position = current_position - unit_vector * step
        new_orientation = current_orientation
        fly_type = "move"

    elif action == 'right':
        unit_x = 1.0 * math.cos(yaw + math.pi / 2)
        unit_y = 1.0 * math.sin(yaw + math.pi / 2)
        vector = np.array([unit_x, unit_y, 0])
        norm = np.linalg.norm(vector)
        unit_vector = vector / norm if norm > 1e-6 else np.array([0, 0, 0])
        step = AirsimActionSettings.LEFT_RIGHT_STEP_SIZE if is_fixed else step_size
        new_position = current_position + unit_vector * step
        new_orientation = current_orientation
        fly_type = "move"

    else:  # stop
        new_position = current_position
        new_orientation = current_orientation
        fly_type = "stop"

    return new_position, new_orientation, fly_type