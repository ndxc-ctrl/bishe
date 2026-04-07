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
    # 【新增航线领航】：要求千问在无异物时报告公路位置走向
    system_prompt = f"""
            你是一个专业的无人机公路巡检视觉助手。用户会提供 {n} 张无人机视角的图片（包含前、左、右、下四个视角，其中最后一张通常是俯视图）。
            你的任务是识别公路上是否有异常情况，并在正常飞行时协助保持航线居中。

            【严格判定标准】：
            1. 异物必须存在于公路路面上！
            2. 正常行驶的汽车、路边的树木、草坪、阴影、地面标线绝对不是异物！
            3. 如果发现异常（如掉落的货物箱、落石、事故车等），即为有异物。

            请针对每一张图片输出一句详细的语义描述（绝对不能换行），并在结尾明确加上结论：
            1. 如果没有发现异物，必须判断当前公路主体在画面中的位置走向（偏左、偏右还是居中）。
               结尾格式必须为：“结论：公路无异物，公路位置[偏左/偏右/居中]”
            2. 如果确信发现路面异物，必须在结尾加上：“结论：公路有异物，是[具体异物描述]”

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
            return ["未能看清路面。结论：公路无异物，公路位置居中"] * n

        raw = response["output"]["choices"][0]["message"].content[0]["text"].strip()

        # 兼容部分情况下的 json 格式符
        if raw.startswith("```json"):
            raw = raw[7:-3].strip()

        captions = json.loads(raw)
        if len(captions) != n:
            return ["未能看清路面。结论：公路无异物，公路位置居中"] * n

        return captions

    except Exception as e:
        print(f">>> [视觉拦截] API 发生异常: {e}，启用兜底机制。")
        return ["API调用异常。结论：公路无异物，公路位置居中"] * n