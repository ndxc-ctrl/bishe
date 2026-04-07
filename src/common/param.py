import os
import sys
import argparse
import datetime
from pathlib import Path

# ==============================================================
# 定位到根目录，洗掉假缓存
# ==============================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_ROOT_DIR = os.path.abspath(os.path.join(_SRC_DIR, ".."))

if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

for k in list(sys.modules.keys()):
    if k == 'utils' or k.startswith('utils.'):
        del sys.modules[k]

# 恢复使用前缀！正常导入！
from uav_utils.CN import CN

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ("yes", "true", "t", "1"): return True
    elif v.lower() in ("no", "false", "f", "0"): return False
    else: raise argparse.ArgumentTypeError("Boolean value expected.")

class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="UAV_Search")
        self.parser.add_argument("--project_prefix", type=str, default=str(_ROOT_DIR))
        self.parser.add_argument("--name", type=str, default="ON-Air")
        self.parser.add_argument("--maxActions", type=int, default=150)
        self.parser.add_argument("--xOy_step_size", type=int, default=500)
        self.parser.add_argument("--z_step_size", type=int, default=200)
        self.parser.add_argument("--rotateAngle", type=int, default=15)
        self.parser.add_argument("--batchSize", type=int, default=1)
        self.parser.add_argument("--simulator_tool_port", type=int, default=30000)
        self.parser.add_argument("--dataset_path", type=str, default='./DATASET/uavondataset.json')
        self.parser.add_argument("--is_fixed", type=str2bool, default=False)
        self.parser.add_argument("--gpu_id", type=int, default=0)
        self.parser.add_argument("--generation_model_path", type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
        self.parser.add_argument("--eval_save_path", type=str, default='./unfixed_logs/scene')
        self.args = self.parser.parse_args()

parm = Param()
args = parm.args
args.make_dir_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
args.logger_file_name = os.path.join(args.project_prefix, 'DATA', 'output', args.name, 'logs', f'{args.name}_{args.make_dir_time}.log')
args.machines_info = [{'MACHINE_IP': '127.0.0.1', 'SOCKET_PORT': int(args.simulator_tool_port), 'MAX_SCENE_NUM': 16, 'open_scenes': []}]

default_config = CN.clone()
default_config.make_dir_time = args.make_dir_time
try:
    default_config.freeze()
except: pass