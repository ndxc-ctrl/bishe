import math


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