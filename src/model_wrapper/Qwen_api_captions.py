import base64
from io import BytesIO
import traceback
from PIL import Image
import numpy as np
import json
import time
from dashscope import MultiModalConversation


def encode_image(image_files):
    base_img = []
    for file in image_files:
        if isinstance(file, np.ndarray):
            file = np.ascontiguousarray(file)
            if file.dtype != np.uint8:
                file = file.astype(np.uint8)
            img = Image.fromarray(file)
        elif isinstance(file, Image.Image):
            img = file
        else:
            raise TypeError(f"不支持的图像数据类型: {type(file)}")

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base_img.append(encoded_string)
    return base_img


def generate_caption(image_file, temperature=0.7):
    n = len(image_file)
    system_prompt = f"""
            你是一个专业的无人机区域巡检视觉助手。用户会提供 {n} 张无人机视角的图片（包含前、左、右、下四个视角，其中最后一张通常是俯视图）。
            你的任务是全面扫描这片区域，仔细识别画面中是否存在任何【异常情况】。

            【严格判定标准】：
            1. 正常行驶的汽车、路边的树木、草坪、阴影、地面标线、普通建筑绝对不是异常！
            2. 只有发现明显的非正常状况（如火灾、散落的障碍物、事故等），才判定为有异常。

            请针对每一张图片输出一句详细的语义描述（绝对不能换行），并在结尾明确加上结论：
            1. 如果没有发现异常，结尾格式必须为：“结论：区域内无异常，正常巡检”
            2. 如果确信发现区域内有异常，必须在结尾加上：“结论：区域内有异常，是[具体异常描述]”

            要求：必须严格返回一个 JSON 数组，数组长度必须刚好是 {n}，数组元素为单行字符串。
    """

    captions = []
    user_content = []
    for img_b64 in image_file:
        user_content.append({'image': 'data:image/png;base64,' + img_b64})
    messages = [{"role": "system", "content": [system_prompt]},
                {'role': 'user', 'content': user_content}]

    try:
        response = MultiModalConversation.call(
            api_key="sk-528d0d3ce55f4c5f830dc00df4733eb5",
            model='qwen-vl-max',
            messages=messages
        )

        if not response or "output" not in response or response["output"] is None:
            return ["未能看清地面。结论：区域内无异常，正常巡检"] * n

        raw = response["output"]["choices"][0]["message"].content[0]["text"].strip()

        # 兼容部分情况下的 json 格式符
        if raw.startswith("```json"):
            raw = raw[7:-3].strip()

        captions = json.loads(raw)
        if len(captions) != n:
            return ["未能看清地面。结论：区域内无异常，正常巡检"] * n

        return captions

    except Exception as e:
        print(f">>> [视觉拦截] API 发生异常: {e}，启用兜底机制。")
        return ["API调用异常。结论：区域内无异常，正常巡检"] * n