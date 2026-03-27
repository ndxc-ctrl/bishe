import os
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
import sys
import time
from pathlib import Path
import cv2  # 新增：用于视频录制
from PIL import Image  # 新增：用于图像处理

# ==============================================================
# 第一步：把项目根目录置顶
# ==============================================================
SRC_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, ROOT_DIR)

if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]

from omni.isaac.kit import SimulationApp

config = {
    "headless": True,
    "active_gpu": 0,
    "physics_gpu": 0,
    "multi_gpu": False,
    "livestream": 2
}
simulation_app = SimulationApp(config)

# ==============================================================
# 第二步：核心清理！引擎启动后清洗被污染的 utils，并重申控制权
# ==============================================================
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, ROOT_DIR)

for k in list(sys.modules.keys()):
    if k == 'utils' or k.startswith('utils.'):
        del sys.modules[k]

# ==============================================================
# 第三步：恢复带前缀的正常导入！
# ==============================================================
import torch
import tqdm
import numpy as np
import random

from common.param import args
from env_uav import AirVLNENV
from src.closeloop_util import BatchIterator, EvalBatchState
from uav_utils.logger import logger
from model_wrapper.base_model import BaseModelWrapper
from model_wrapper.ON_Air_2 import ONAir


def eval(modelWrapper: BaseModelWrapper, env: AirVLNENV, is_fixed, save_eval_path):
    # --- 新增：初始化视频写入器 ---
    video_path = os.path.join(save_eval_path, "patrol_demo.mp4")
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 2.0, (640, 480))

    with torch.no_grad():
        data = BatchIterator(env)
        data_len = len(data)
        pbar = tqdm.tqdm(total=data_len, desc="batch")
        cnt = 0
        while True:
            env_batch = env.next_minibatch(skip_scenes=[])
            if env_batch is None: break
            batch_state = EvalBatchState(batch_size=env.batch_size, env_batchs=env_batch, env=env,
                                         save_eval_path=save_eval_path)
            pbar.update(n=env.batch_size)
            inputs, user_prompts = modelWrapper.prepare_inputs(batch_state.episodes, is_fixed)
            cnt += env.batch_size

            for i in range(env.batch_size):
                print("\n" + "*" * 50)
                print(f"[任务启动] 任务号: {batch_state.task_id[i]}")
                print(f"1. [用户指令]: {user_prompts[i]}")
                init_pos = batch_state.episodes[i][0]['sensors']['state']['position']
                print(f"2. [起始位置]: {init_pos}")
                print("*" * 50 + "\n")

            for t in range(args.maxActions):
                logger.info('Step: {} \t Completed: {} / {}'.format(t, cnt - batch_state.skips.count(False), data_len))
                start1 = time.time()
                actions, steps_size, dones = modelWrapper.run(inputs, is_fixed)
                print("get actions time:", time.time() - start1)

                # --- 新增：获取当前画面的RGB并写入视频 ---
                if hasattr(batch_state, 'episodes') and len(batch_state.episodes) > 0 and len(
                        batch_state.episodes[0]) > 0:
                    # 尝试提取前置相机的 RGB 图像
                    current_rgb = batch_state.episodes[0][-1].get('rgb', [None])[0]
                    if current_rgb is not None:
                        if isinstance(current_rgb, Image.Image):
                            frame = np.array(current_rgb)
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        else:
                            frame = np.array(current_rgb)
                            if len(frame.shape) == 3 and frame.shape[2] == 3:
                                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                        frame = cv2.resize(frame, (640, 480))
                        action_str = actions[0] if actions else "None"
                        cv2.putText(frame, f"Step: {t} | Action: {action_str}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 2)

                        # 如果发现异常，则用红色字体打上警告，并多写入几帧
                        if dones[0] or actions[0] == 'stop':
                            cv2.putText(frame, "ANOMALY FOUND! STOPPED.", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 3)
                            for _ in range(5):
                                video_writer.write(frame)
                        else:
                            video_writer.write(frame)
                # -----------------------------------------------------

                for i in range(env.batch_size):
                    if dones[i]: batch_state.dones[i] = True
                for i in range(len(actions)):
                    print(actions[i], ":", steps_size[i])

                env.makeActions(actions, steps_size, is_fixed)
                obs = env.get_obs()
                batch_state.update_from_env_output(obs, user_prompts, actions, steps_size, is_fixed)

                for i in range(env.batch_size):
                    if not batch_state.skips[i]:
                        new_pos = batch_state.episodes[i][-1]['sensors']['state']['position']
                        actual_step_size = batch_state.steps_size[i][-1]
                        print(
                            f"  -> [步骤 {t}] 无人机 {i} | 执行动作: {actions[i]}, 步长/角度: {actual_step_size} | 新位置: {new_pos}")

                batch_state.update_metric()
                if batch_state.check_batch_termination(t): break
                inputs, user_prompts = modelWrapper.prepare_inputs(batch_state.episodes, is_fixed)

        try:
            pbar.close()
        except:
            pass

        # --- 新增：释放视频资源 ---
        video_writer.release()
        print(f"\n====== 巡检演示视频已成功保存至: {video_path} ======\n")


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if args.dataset_path.startswith("./"):
        args.dataset_path = os.path.join(ROOT_DIR, args.dataset_path.replace("./", ""))

    env = AirVLNENV(
        batch_size=args.batchSize,
        dataset_path=args.dataset_path,
        save_path=args.eval_save_path,
        seed=seed
    )
    fixed = args.is_fixed
    save_eval_path = os.path.join(args.eval_save_path, args.name)
    if not os.path.exists(save_eval_path): os.makedirs(save_eval_path)

    modelWrapper = ONAir(fixed=fixed, batch_size=args.batchSize)
    eval(modelWrapper=modelWrapper, env=env, is_fixed=fixed, save_eval_path=save_eval_path)
    env.delete_VectorEnvUtil()