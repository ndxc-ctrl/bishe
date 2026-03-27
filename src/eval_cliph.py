import torch
import tqdm
import os
from pathlib import Path
import sys
import time
import random
import numpy as np 
sys.path.append(str(Path(str(os.getcwd())).resolve()))
from common.param import args
from env_uav import AirVLNENV
from src.closeloop_util import BatchIterator, EvalBatchState, initialize_env_eval
from uav_utils.logger import logger
from model_wrapper.base_model import BaseModelWrapper
from model_wrapper.CLIP_H import CLIP_H

def eval(modelWrapper: BaseModelWrapper, env: AirVLNENV ,is_fixed, save_eval_path):
    with torch.no_grad():
        data = BatchIterator(env)
        data_len = len(data)
        pbar = tqdm.tqdm(total=data_len,desc="batch")
        cnt=0
        while True:
           
            env_batch = env.next_minibatch(skip_scenes=[])
            
            if env_batch is None:
                break
            
            batch_state = EvalBatchState(batch_size=env.batch_size, env_batchs=env_batch, env=env, save_eval_path= save_eval_path)
          
            pbar.update(n = env.batch_size)

            inputs, user_prompts, depths = modelWrapper.prepare_inputs(batch_state.episodes)
            cnt += env.batch_size

            # ================= [新增打印 1 & 2: 初始指令与起始位置] =================
            for i in range(env.batch_size):
                print("\n" + "*" * 50)
                print(f"[任务启动] 任务号: {batch_state.task_id[i]}")
                print(f"1. [用户指令]: {user_prompts[i]}")
                init_pos = batch_state.episodes[i][0]['sensors']['state']['position']
                print(f"2. [起始位置]: {init_pos}")
                print("*" * 50 + "\n")
            # ========================================================================

            for t in range(args.maxActions):
                logger.info('Step: {} \t Completed: {} / {}'.format(t, cnt-batch_state.skips.count(False), data_len))
          
                start1 = time.time()
                actions, steps_size, dones = modelWrapper.run(inputs, depths)
                print("get actions time:",time.time()-start1)

                import cv2
                import airsim
                for i in range(env.batch_size):
                    if dones[i] and not batch_state.skips[i]:
                        batch_state.dones[i] = True

                        # ================= [修正逻辑: 任务结束保存俯视图] =================
                        try:
                            # 1. 从 env.simulator_tool 中拍平所有的 airsim_clients
                            all_clients = []
                            for row in env.simulator_tool.airsim_clients:
                                all_clients.extend(row)

                            # 2. 根据当前 batch 索引 i 拿到对应的真实 client
                            client = all_clients[i]

                            if client is not None:
                                # 调用 3 号相机（通常是下视/俯视相机）拍摄照片
                                responses = client.simGetImages(
                                    [airsim.ImageRequest("3", airsim.ImageType.Scene, False, False)])
                                if responses and len(responses[0].image_data_uint8) > 0:
                                    img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                                    img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)

                                    # 将图像保存到当前目录
                                    save_name = f"TopDownView_Task_{batch_state.task_id[i]}.png"
                                    cv2.imwrite(save_name, img_rgb)
                                    print(
                                        f"\n📸 [成功] 任务 {batch_state.task_id[i]} 结束，已保存目标俯视图: {save_name}\n")
                        except Exception as e:
                            print(f"\n[警告] 任务 {batch_state.task_id[i]} 保存俯视图失败: {e}\n")
                        # ===================================================================

                for i in range(len(actions)):
                    print(actions[i],":",steps_size[i])
                env.makeActions(actions, steps_size, is_fixed)
                
                ###get next step observations
                ###get next step observations
                obs = env.get_obs()
                batch_state.update_from_env_output(obs, user_prompts, actions, steps_size, is_fixed)

                # ================= [新增打印 3: 每步执行的操作与新位置] =================
                for i in range(env.batch_size):
                    if not batch_state.skips[i]:  # 如果该无人机的任务还没结束
                        new_pos = batch_state.episodes[i][-1]['sensors']['state']['position']
                        actual_step_size = batch_state.steps_size[i][-1]
                        print(
                            f"  -> [步骤 {t}] 无人机 {i} | 执行动作: {actions[i]}, 步长: {actual_step_size} | 新位置: {new_pos}")
                # ========================================================================

                batch_state.update_metric()
                is_terminate = batch_state.check_batch_termination(t)
                if is_terminate:
                    break      
                inputs, user_prompts, depths = modelWrapper.prepare_inputs(batch_state.episodes)
        try:
            pbar.close()
        except:
            pass


if __name__ == "__main__":
    seed = 42  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    env = initialize_env_eval(dataset_path=args.dataset_path, save_path=args.eval_save_path)
    fixed = args.is_fixed

    save_eval_path = os.path.join(args.eval_save_path, args.name)
    if not os.path.exists(args.eval_save_path):
        os.makedirs(args.eval_save_path)

    modelWrapper = CLIP_H(batch_size=args.batchSize)

    eval(modelWrapper=modelWrapper, env=env, is_fixed=fixed, save_eval_path=save_eval_path)

    env.delete_VectorEnvUtil()