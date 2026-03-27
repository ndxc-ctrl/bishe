import math
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R

from src.common.param import args
from uav_utils.logger import logger

# ==========================================
# 直接在这里定义固定模式下的默认步长常量
# 彻底摆脱对 airsim_plugin 文件夹的依赖
# ==========================================
class AirsimActionSettings:
    FORWARD_STEP_SIZE = 5.0      # 默认前进 5 米
    LEFT_RIGHT_STEP_SIZE = 5.0   # 默认左右飞 5 米
    UP_DOWN_STEP_SIZE = 2.0      # 默认升降 2 米
    TURN_ANGLE = 15.0            # 默认旋转 15 度

class SimState:
    def __init__(self, index=-1,
                 step=0,
                 task_info ={}
                 ):
# ... 后面保持不变 ...
        self.index = index
        self.step = step
        self.task_info = task_info
        self.is_end = False
        self.oracle_success = False
        self.is_collisioned = False
        self.predict_start_index = 0
        self.history_start_indexes = [0]
        self.SUCCESS_DISTANCE = 20
        self.progress = 0.0
        self.waypoint = {}
        self.sensorInfo = {}
        self.target_position = task_info['object_position'][-1]
        self.start_pose = task_info['start_pose']

        # 统一使用列表 [x, y, z] 和 [x, y, z, w]
        self.trajectory = [{'sensors': {
            'state': {
                'position': self.start_pose['start_position'],
                'quaternionr': self.start_pose['start_quaternionr']
            },
        },
            'move_distance': float(0.0),
            'distance_to_target': self.task_info['distance_to_target']
        }]
        self.move_distance = 0.0
        self.heading_changes: list[float] = []

    @property
    def state(self):
        return self.trajectory[-1]['sensors']['state']

    @property
    def pose(self):
        # 返回拼接的 list: [x, y, z, x, y, z, w]
        return self.trajectory[-1]['sensors']['state']['position'] + self.trajectory[-1]['sensors']['state'][
            'quaternionr']


class ENV:
    def __init__(self, load_scenes: list):
        self.batch = None

    def set_batch(self, batch):
        self.batch = copy.deepcopy(batch)
        return

    def get_obs_at(self, index: int, state: SimState):
        assert self.batch is not None, 'batch is None'
        item = self.batch[index]
        oracle_success = state.oracle_success
        done = state.is_end
        return (done, oracle_success), state


def getNextPosition(current_position, current_orientation, action, step_size, is_fixed):
    """
    替换了原本的 airsim.Pose 参数。
    current_position: list or np.array [x, y, z]
    current_orientation: list or np.array [x, y, z, w]
    """
    current_position = np.array(current_position)

    # 欧拉角转换 (scipy 默认的 euler 返回也是 roll, pitch, yaw 顺序，取决于指定的轴序)
    # Airsim 默认是 NED 坐标系, Isaac 通常是 ENU。此处保留原有数学逻辑。
    r = R.from_quat(current_orientation)
    euler = r.as_euler('xyz', degrees=False)  # roll, pitch, yaw
    roll, pitch, yaw = euler[0], euler[1], euler[2]

    if action == 'forward':
        dx = math.cos(yaw)
        dy = math.sin(yaw)
        dz = 0

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
        if math.degrees(new_yaw) < -180:
            new_yaw += math.radians(360)

        new_position = current_position
        new_orientation = R.from_euler('xyz', [roll, pitch, new_yaw]).as_quat()
        fly_type = "rotate"

    elif action == 'rotr':
        step = AirsimActionSettings.TURN_ANGLE if is_fixed else step_size
        new_yaw = yaw + math.radians(step)
        if math.degrees(new_yaw) > 180:
            new_yaw += math.radians(-360)

        new_position = current_position
        new_orientation = R.from_euler('xyz', [roll, pitch, new_yaw]).as_quat()
        fly_type = "rotate"

    elif action == 'ascend':
        unit_vector = np.array([0, 0, -1])  # 如果 Isaac是 Z朝上，这里可能需要改为 [0, 0, 1]，视后续调试而定
        step = AirsimActionSettings.UP_DOWN_STEP_SIZE if is_fixed else step_size
        new_position = current_position + unit_vector * step
        new_orientation = current_orientation
        fly_type = "move"

    elif action == 'descend':
        unit_vector = np.array([0, 0, 1])
        step = AirsimActionSettings.UP_DOWN_STEP_SIZE if is_fixed else step_size
        new_position = current_position + unit_vector * step
        new_orientation = current_orientation
        fly_type = "move"

    elif action == 'left':
        unit_x = 1.0 * math.cos(math.radians(float(yaw * 180 / math.pi) + 90))
        unit_y = 1.0 * math.sin(math.radians(float(yaw * 180 / math.pi) + 90))
        vector = np.array([unit_x, unit_y, 0])

        norm = np.linalg.norm(vector)
        unit_vector = vector / norm if norm > 1e-6 else np.array([0, 0, 0])

        step = AirsimActionSettings.LEFT_RIGHT_STEP_SIZE if is_fixed else step_size
        new_position = current_position - unit_vector * step
        new_orientation = current_orientation
        fly_type = "move"

    elif action == 'right':
        unit_x = 1.0 * math.cos(math.radians(float(yaw * 180 / math.pi) + 90))
        unit_y = 1.0 * math.sin(math.radians(float(yaw * 180 / math.pi) + 90))
        vector = np.array([unit_x, unit_y, 0])

        norm = np.linalg.norm(vector)
        unit_vector = vector / norm if norm > 1e-6 else np.array([0, 0, 0])

        step = AirsimActionSettings.LEFT_RIGHT_STEP_SIZE if is_fixed else step_size
        new_position = current_position + unit_vector * step
        new_orientation = current_orientation
        fly_type = "move"
    else:
        new_position = current_position
        new_orientation = current_orientation
        fly_type = "stop"

    return new_position, new_orientation, fly_type