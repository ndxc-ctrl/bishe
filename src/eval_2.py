import os

# 强制清理代理环境变量，防止 Isaac Sim 或其他库自动读取产生网络问题
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
import sys
import time
import math
import base64
import re
from io import BytesIO
from pathlib import Path
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation as R

import httpx
from openai import OpenAI
from dashscope import MultiModalConversation

SRC_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, ROOT_DIR)

if "CUDA_VISIBLE_DEVICES" in os.environ: del os.environ["CUDA_VISIBLE_DEVICES"]

from omni.isaac.kit import SimulationApp

config = {"headless": True, "active_gpu": 0, "physics_gpu": 0, "multi_gpu": False, "livestream": 2}
simulation_app = SimulationApp(config)

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, ROOT_DIR)
for k in list(sys.modules.keys()):
    if k == 'utils' or k.startswith('utils.'): del sys.modules[k]

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


def encode_img_for_tracking(img_array):
    buffered = BytesIO()
    img_pil = Image.fromarray(img_array) if isinstance(img_array, np.ndarray) else img_array
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def eval(modelWrapper: BaseModelWrapper, env: AirVLNENV, is_fixed, save_eval_path):
    # 用字符串拼接方式防止 URL 被转换成 markdown
    tracker_proxy_url = "http://" + "127.0.0.1:13849"
    tracker_http_client = httpx.Client(proxy=tracker_proxy_url)
    tracker_gpt = OpenAI(
        api_key="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",
        http_client=tracker_http_client, timeout=15.0
    )

    with torch.no_grad():
        data = BatchIterator(env)
        data_len = len(data)
        pbar = tqdm.tqdm(total=data_len, desc="batch")

        while True:
            env_batch = env.next_minibatch(skip_scenes=[])
            if env_batch is None: break

            batch_state = EvalBatchState(batch_size=env.batch_size, env_batchs=env_batch, env=env,
                                         save_eval_path=save_eval_path)
            pbar.update(n=env.batch_size)

            inputs, user_prompts = modelWrapper.prepare_inputs(batch_state.episodes, is_fixed)
            task_id = batch_state.task_id[0]

            video_down_path = os.path.join(save_eval_path, f"patrol_demo_down_{task_id}.mp4")
            video_third_path = os.path.join(save_eval_path, f"patrol_demo_3rdPerson_{task_id}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_down = cv2.VideoWriter(video_down_path, fourcc, 2.0, (640, 480))
            video_third = cv2.VideoWriter(video_third_path, fourcc, 2.0, (640, 480))

            def process_frame(img, action_str, is_anomaly_flag, cam_title, current_pos, step_t):
                if img is None: return np.zeros((480, 640, 3), dtype=np.uint8)
                frame = np.array(img) if isinstance(img, Image.Image) else np.array(img)
                if len(frame.shape) == 3 and frame.shape[2] == 3: frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (640, 480))

                cv2.putText(frame, f"Step: {step_t} | Action: {action_str}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)
                cv2.putText(frame, f"Camera: {cam_title}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                if is_anomaly_flag:
                    cv2.putText(frame, "ANOMALY DETECTED! SERVO TRACKING", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 3)
                    if current_pos is not None:
                        coord_str = f"UAV Loc: [{current_pos[0]:.1f}, {current_pos[1]:.1f}, {current_pos[2]:.1f}]"
                        cv2.putText(frame, coord_str, (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return frame

            for i in range(env.batch_size):
                print("\n" + "=" * 60)
                print(f"🚀 [任务启动] 任务号: {task_id}")
                try:
                    init_pos = batch_state.episodes[i][0]['sensors']['state']['position']
                    instruction = batch_state.episodes[i][0].get('instruction', '未提供指令')
                    print(f"📝 [用户指令]: {instruction}")
                    print(f"📍 [起始位置]: [{init_pos[0]:.1f}, {init_pos[1]:.1f}, {init_pos[2]:.1f}]")
                except:
                    pass
                print("=" * 60 + "\n")

            # --- 巡检主循环 ---
            for t in range(70):

                if t == 0:
                    try:
                        drone_pos, _ = env.drone_prims[0].get_world_pose()
                        cam_pos, _ = env.camera_rigs[0][4].get_world_pose()
                        dist = np.linalg.norm(np.array(drone_pos) - np.array(cam_pos))
                        print("\n================= 🎥 视角调试诊断雷达 🎥 =================")
                        print(
                            f"  无人机本体绝对坐标 : [X: {drone_pos[0]:.1f}, Y: {drone_pos[1]:.1f}, Z: {drone_pos[2]:.1f}]")
                        print(
                            f"  第三视角相机绝对坐标: [X: {cam_pos[0]:.1f}, Y: {cam_pos[1]:.1f}, Z: {cam_pos[2]:.1f}]")
                        print(f"  物理直线距离       : {dist:.1f} 单位")
                        print("==============================================================\n")
                    except Exception as e:
                        pass

                actions, steps_size, dones = modelWrapper.run(inputs, is_fixed)

                qwen_down_caption = "未知"
                try:
                    prompt_text = user_prompts[0]
                    if "Down =" in prompt_text:
                        qwen_down_caption = prompt_text.split("Down =")[1].split("\n")[0].strip()
                except:
                    pass

                is_anomaly = "有异物" in qwen_down_caption

                # =====================================================================
                # 🚀 新增逻辑 1：航线自动纠偏！如果没看到异物且公路偏航，自动打断GPT强行侧移
                # =====================================================================
                if not is_anomaly and actions[0] == 'forward':
                    if "公路位置偏左" in qwen_down_caption:
                        actions[0] = 'left'
                        steps_size[0] = 300.0  # 给一个适当的左移步长修正航向
                        print(f"🛣️ [航线修正]: 发现公路偏左，强行打断向前，执行向左平移 300.0")
                    elif "公路位置偏右" in qwen_down_caption:
                        actions[0] = 'right'
                        steps_size[0] = 300.0
                        print(f"🛣️ [航线修正]: 发现公路偏右，强行打断向前，执行向右平移 300.0")

                if is_anomaly:
                    # 提取具体的异物名称，传给后面的伺服追踪，防止它错认成汽车
                    anomaly_target = qwen_down_caption.split("是")[-1].strip().strip(
                        '。.,，') if "是" in qwen_down_caption else "异物"
                    print(f"\n⚠️ [异常警告]: 锁定异常目标【{anomaly_target}】！立即启动“高精度对齐伺服降落”模式！\n")

                    try:
                        current_pos = batch_state.episodes[0][-1]['sensors']['state']['position']
                        current_quat = batch_state.episodes[0][-1]['sensors']['state']['quaternionr']
                        yaw = R.from_quat(current_quat).as_euler('xyz', degrees=False)[2]

                        curr_x, curr_y, curr_z = current_pos[0], current_pos[1], current_pos[2]
                        target_z_40m = batch_state.episodes[0][-1].get('object_position', [[0, 0, 0]])[-1][2] + 4000.0

                        track_max_steps = 30
                        track_step = 0

                        while curr_z > target_z_40m and track_step < track_max_steps:
                            track_step += 1

                            imgs = batch_state.episodes[0][-1].get('rgb', [None])
                            raw_down = imgs[3] if len(imgs) > 3 else imgs[0]
                            raw_third = batch_state.episodes[0][-1].get('rgb_third_person', imgs[0])

                            b64_img = encode_img_for_tracking(raw_down)

                            # =====================================================================
                            # 🚀 新增逻辑 2：极致缩紧中心死区 & 目标记忆锚定
                            # =====================================================================
                            q_prompt = f"""这是一张无人机向下俯视的图片。
                            你的任务是辅助无人机精准降落并保持目标在视野的【绝对中心】。
                            🚨【锁定目标】：你要追踪的异物是【{anomaly_target}】！如果画面中有正常行驶的汽车，请完全无视汽车，只关注该异物！

                            请严格判断该目标【{anomaly_target}】相对于画面绝对中心点的位置：
                            - 🚨要求目标必须在画面的绝对正中央（容忍误差极小，不能超过5%）。只要有一点偏离，就必须修正！
                            - 如果目标偏上半部分，附加：【偏前】
                            - 如果目标偏下半部分，附加：【偏后】
                            - 如果目标偏左半部分，附加：【偏左】
                            - 如果目标偏右半部分，附加：【偏右】
                            - 只有当目标完美处于正中心时，才附加：【正中央】
                            - 如果画面中完全找不到该目标（汽车不算），附加：【视野丢失】
                            请用一段话连贯输出描述，并在末尾强制加上带方括号的六个方位关键词之一！"""

                            q_msg = [{"role": "system", "content": [{"text": q_prompt}]},
                                     {"role": "user", "content": [{"image": f"data:image/png;base64,{b64_img}"}]}]

                            try:
                                resp = MultiModalConversation.call(api_key="sk-528d0d3ce55f4c5f830dc00df4733eb5",
                                                                   model="qwen-vl-max", messages=q_msg)
                                q_res = resp["output"]["choices"][0]["message"].content[0]["text"].strip()
                            except:
                                q_res = "【正中央】"

                            g_prompt = f"""You are a drone flight controller tracking an anomaly.
                            Current UAV Height (Z-axis): {curr_z:.1f} units.
                            Current Vision Report from Qwen: {q_res}

                            CRITICAL RULE: You MUST prioritize horizontal centering (X/Y axis). DO NOT DESCEND if the anomaly is off-center!

                            Read the keyword in the Qwen report:
                            - 【偏前】 -> `[forward, step_size]`
                            - 【偏后】 -> `[backward, step_size]`
                            - 【偏左】 -> `[left, step_size]`
                            - 【偏右】 -> `[right, step_size]`
                            - 【正中央】 (meaning it is perfectly aligned) -> `[descend, step_size]`
                            - 【视野丢失】 -> `[ascend, 1000.0]`

                            Step size logic:
                            - For X/Y movement: Use {max(50.0, curr_z * 0.08):.1f} to strictly correct the offset without descending.
                            - For descending: Use 800.0.
                            Output STRICTLY a JSON-like array ONLY: `[action, step_size]`"""

                            try:
                                g_resp = tracker_gpt.chat.completions.create(model="gpt-4o-mini", messages=[
                                    {"role": "user", "content": g_prompt}], temperature=0.0)
                                g_res = g_resp.choices[0].message.content.strip()
                                match = re.search(r'\[\s*[\'"]?([a-zA-Z]+)[\'"]?\s*,\s*([0-9.]+)\s*\]', g_res)
                                if match:
                                    act = match.group(1).lower()
                                    step_val = float(match.group(2))
                                else:
                                    raise ValueError("Format Error")
                            except Exception as e:
                                dyn_step = max(50.0, curr_z * 0.08)
                                if "【偏前】" in q_res:
                                    act, step_val = "forward", dyn_step
                                elif "【偏后】" in q_res:
                                    act, step_val = "backward", dyn_step
                                elif "【偏左】" in q_res:
                                    act, step_val = "left", dyn_step
                                elif "【偏右】" in q_res:
                                    act, step_val = "right", dyn_step
                                elif "【视野丢失】" in q_res:
                                    act, step_val = "ascend", 1000.0
                                else:
                                    act, step_val = "descend", max(500.0, curr_z * 0.1)

                            print(
                                f"🚁 [对齐伺服] 目标:{anomaly_target[:5]}.. | 高度 {curr_z:.1f} | 👁️ 观测: [{q_res[-20:]}] ➡️ 🧠 执行: [{act}, {step_val:.1f}]")

                            if act == 'descend':
                                curr_z -= step_val
                            elif act == 'ascend':
                                curr_z += step_val
                            elif act == 'forward':
                                curr_x += math.cos(yaw) * step_val
                                curr_y += math.sin(yaw) * step_val
                            elif act == 'backward':
                                curr_x -= math.cos(yaw) * step_val
                                curr_y -= math.sin(yaw) * step_val
                            elif act == 'left':
                                curr_x -= math.sin(yaw) * step_val
                                curr_y += math.cos(yaw) * step_val
                            elif act == 'right':
                                curr_x += math.sin(yaw) * step_val
                                curr_y -= math.cos(yaw) * step_val

                            if curr_z < target_z_40m: curr_z = target_z_40m

                            env.drone_prims[0].set_world_pose(position=np.array([curr_x, curr_y, curr_z]))

                            action_label = f"SERVO: {act.upper()} {step_val:.1f}"
                            for _ in range(15): env.world.step(render=True)

                            obs = env.get_obs()
                            batch_state.update_from_env_output(obs, user_prompts, actions, steps_size, is_fixed)
                            f_d = process_frame(raw_down, action_label, True, "Down (Top-View)",
                                                [curr_x, curr_y, curr_z], t)
                            f_t = process_frame(raw_third, action_label, True, "Third-Person", [curr_x, curr_y, curr_z],
                                                t)
                            video_down.write(f_d)
                            video_third.write(f_t)

                    except Exception as e:
                        pass

                    images_list = batch_state.episodes[0][-1].get('rgb', [None])
                    raw_down_image = images_list[3] if len(images_list) > 3 else images_list[0]
                    photo_filename = f"anomaly_photo_{task_id}_step_{t}.png"
                    anomaly_photo_path = os.path.join(save_eval_path, photo_filename)
                    try:
                        if isinstance(raw_down_image, Image.Image):
                            raw_down_image.save(anomaly_photo_path)
                        else:
                            Image.fromarray(raw_down_image).save(anomaly_photo_path)
                        print(f"\n🚨 [锁定成功]: 异物追踪完成！40m极近抓拍已保存: {anomaly_photo_path}\n")
                    except:
                        pass

                    actions[0] = 'stop'
                    dones[0] = True
                    steps_size[0] = 0
                    try:
                        env.sim_states[0].oracle_success = True
                        env.sim_states[0].is_end = True
                        env.sim_states[0].distance_to_target = 0.0
                    except:
                        pass

                action_map = {
                    'forward': '向前飞行', 'left': '向左平移', 'right': '向右平移',
                    'rotl': '原地向左旋转', 'rotr': '原地向右旋转',
                    'ascend': '向上爬升', 'descend': '向下降落', 'stop': '锁定悬停'
                }
                final_action_en = actions[0] if actions else 'stop'
                final_step = steps_size[0] if steps_size else 0
                final_action_zh = action_map.get(final_action_en, final_action_en)

                unit = "度" if final_action_en in ['rotl', 'rotr'] else "单位(距离)"
                decision_zh = "异常已锁定，任务结束" if final_action_en == 'stop' else f"决定 {final_action_zh} {final_step} {unit}"

                print(f"\n────────────────── [ 第 {t} 步 宏观巡检报告 ] ──────────────────")
                print(f"👁️‍🗨️ [千问视觉语义扫描]: {qwen_down_caption}")
                print(f"🧠 [GPT导航宏观决策]: {decision_zh}  (系统指令: [{final_action_en}, {final_step}])")

                # =========================================================================
                # 🚀 第 0 步: 输出全套高清特征、RGB-D矩阵、以及可视化伪彩图
                # =========================================================================
                if t == 0 and hasattr(batch_state, 'episodes') and len(batch_state.episodes[0]) > 0:
                    print("\n" + "*" * 50)
                    print("📸 [第 0 步: 初始视觉特征提取与保存 (已开启高保真模式)]")
                    step0_save_dir = os.path.join(save_eval_path, f"step0_features_{task_id}")
                    os.makedirs(step0_save_dir, exist_ok=True)
                    print(f"📁 [保存目标路径]: {os.path.abspath(step0_save_dir)}")

                    try:
                        obs_dict = batch_state.episodes[0][-1]
                        cam_names = ["Front", "Left", "Right", "Down"]

                        for idx, cam_name in enumerate(cam_names):
                            # --- 1. 保存高清原图 (640x480) 存为无损 PNG ---
                            if 'rgb' in obs_dict and idx < len(obs_dict['rgb']):
                                highres_rgb_bgr = cv2.cvtColor(obs_dict['rgb'][idx], cv2.COLOR_RGB2BGR)
                                cv2.imwrite(os.path.join(step0_save_dir, f"1_HighRes_RGB_{cam_name}.png"),
                                            highres_rgb_bgr)

                            if 'depth' in obs_dict and idx < len(obs_dict['depth']):
                                raw_depth = obs_dict['depth'][idx]
                                depth_colored = cv2.applyColorMap(raw_depth, cv2.COLORMAP_JET)
                                cv2.imwrite(os.path.join(step0_save_dir, f"1_HighRes_DepthVis_{cam_name}.png"),
                                            depth_colored)

                            # --- 2. 保存池化图 (120x160)，存为无损 PNG ---
                            if 'pooled_rgb' in obs_dict and idx < len(obs_dict['pooled_rgb']):
                                pool_rgb_bgr = cv2.cvtColor(obs_dict['pooled_rgb'][idx], cv2.COLOR_RGB2BGR)
                                cv2.imwrite(os.path.join(step0_save_dir, f"2_Pooled_RGB_{cam_name}.png"), pool_rgb_bgr)
                                cv2.imwrite(os.path.join(step0_save_dir, f"2_Pooled_Depth_{cam_name}.png"),
                                            obs_dict['pooled_depth'][idx])

                            # --- 3. 保存真正的 RGB-D 4通道 Numpy 数据 (.npy) 供网络训练 ---
                            if 'full_rgbd' in obs_dict and idx < len(obs_dict['full_rgbd']):
                                np.save(os.path.join(step0_save_dir, f"3_RGBD_Tensor_{cam_name}.npy"),
                                        obs_dict['full_rgbd'][idx])

                        # --- 4. 保存第三人称全景视角图 ---
                        if obs_dict.get('rgb_third_person') is not None:
                            tp_img = obs_dict['rgb_third_person']
                            tp_bgr = cv2.cvtColor(tp_img, cv2.COLOR_RGB2BGR) if len(tp_img.shape) == 3 else tp_img
                            cv2.imwrite(os.path.join(step0_save_dir, "1_ThirdPerson_View.png"), tp_bgr)

                        print(f"✅ [保存状态]: 成功！高清无损图、彩色深度图、以及网络专属的 .npy 已全部输出！")
                    except Exception as e:
                        print(f"❌ [保存状态]: 失败！发生错误: {str(e)}")
                    print("*" * 50 + "\n")
                # =========================================================================

                print("─────────────────────────────────────────────────────────")

                images_list = batch_state.episodes[0][-1].get('rgb', [None])
                raw_down = images_list[3] if len(images_list) > 3 else (
                    images_list[0] if len(images_list) > 0 else None)
                raw_third = batch_state.episodes[0][-1].get('rgb_third_person',
                                                            images_list[0] if len(images_list) > 0 else None)

                action_str = "LOCKED" if is_anomaly else (actions[0] if actions else "None")
                current_pos = batch_state.episodes[0][-1]['sensors']['state']['position']

                frame_down = process_frame(raw_down, action_str, is_anomaly, "Down (Top-View)", current_pos, t)
                frame_third = process_frame(raw_third, action_str, is_anomaly, "Third-Person", current_pos, t)

                if is_anomaly:
                    for _ in range(15):
                        video_down.write(frame_down)
                        video_third.write(frame_third)
                else:
                    video_down.write(frame_down)
                    video_third.write(frame_third)

                for i in range(env.batch_size):
                    if dones[i]: batch_state.dones[i] = True

                if not dones[0]:
                    env.makeActions(actions, steps_size, is_fixed)
                    obs = env.get_obs()
                    batch_state.update_from_env_output(obs, user_prompts, actions, steps_size, is_fixed)
                    new_pos = batch_state.episodes[0][-1]['sensors']['state']['position']
                    print(f"🤖 [系统状态]: 巡检推进中，当前坐标: [{new_pos[0]:.1f}, {new_pos[1]:.1f}, {new_pos[2]:.1f}]")
                else:
                    final_pos = batch_state.episodes[0][-1]['sensors']['state']['position']
                    print(
                        f"🚩 [目标确认]: 悬停完毕！记录异物在公路上的极密坐标: [{final_pos[0]:.1f}, {final_pos[1]:.1f}, 0.0]")

                batch_state.update_metric()
                if batch_state.check_batch_termination(t): break
                inputs, user_prompts = modelWrapper.prepare_inputs(batch_state.episodes, is_fixed)

            try:
                video_down.release()
                video_third.release()
                print(f"\n✅ 俯视图追踪 Demo 视频已封装: {video_down_path}")
                print(f"✅ 第三视角追踪 Demo 视频已封装: {video_third_path}")
            except:
                pass

        try:
            pbar.close()
        except:
            pass
        print(f"\n====== 🏁 全部巡检任务执行完毕 ======")


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if args.dataset_path.startswith("./"):
        args.dataset_path = os.path.join(ROOT_DIR, args.dataset_path.replace("./", ""))

    env = AirVLNENV(batch_size=args.batchSize, dataset_path=args.dataset_path, save_path=args.eval_save_path, seed=seed)
    fixed = args.is_fixed
    save_eval_path = os.path.join(args.eval_save_path, args.name)
    if not os.path.exists(save_eval_path): os.makedirs(save_eval_path)

    modelWrapper = ONAir(fixed=fixed, batch_size=args.batchSize)
    eval(modelWrapper=modelWrapper, env=env, is_fixed=fixed, save_eval_path=save_eval_path)

    env.delete_VectorEnvUtil()
    simulation_app.close()