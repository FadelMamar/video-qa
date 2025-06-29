# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 17:38:49 2025

@author: FADELCO
"""

import dspy
import os
from abc import abstractmethod, ABC
import base64
from typing import Sequence, List


class Signature(dspy.Signature):
    """Analzye the sequence of images from aerial video surveillance footage and accuractly identify and classify any activity. 
       You should describe accurately what is happening.
       For example, if a person is walking, running, or standing still, if there is a vehicle, a motorcycle doing any activity.
    """

    images: List[dspy.Image] = dspy.InputField(
        desc="sequence of images from videosurveillance camera"
    )
    analysis: str = dspy.OutputField(
        desc="analysis of the sequence of aerial images"
    )

class SummarySignature(dspy.Signature):
    """Analzye the descriptions of videosurveillance images and summarize them in a nice and concise matter for clear decision making.
    """

    image_descriptions: List[str] = dspy.InputField(
        desc="sequence of images from videosurveillance camera"
    )
    summary: str = dspy.OutputField(
        desc="summary of the descriptions"
    )

def load_dspy_model(
    model: str, temperature: float, model_type: str = "chat", cache: bool = True
):
    lm = dspy.LM(
        model,
        api_key=os.environ.get("OPENAI_API_KEY", "sk-no-key-required"),
        temperature=temperature,
        api_base=os.environ.get("BASE_URL"),
        model_type=model_type,
        cache=cache,
    )

    return lm


class Analyzer(ABC):
    @abstractmethod
    def run(*args, **kwargs):
        pass

    def _encode_image(self, image: bytes):
        return base64.b64encode(image).decode("utf-8")

    def format_user_message_images(
        self,
        images: Sequence[bytes],
        prompt: str = "Describe clearly what you see in the sequence of images extracted from a video",
    ):
        base64_frames = [self._encode_image(a) for a in images]
        content = [{"type": "text", "text": prompt}]
        for b64_image in base64_frames:
            base64_qwen = f"data:image/jpeg;base64,{b64_image}"
            content.append({"type": "image_url", "image_url": {"url": base64_qwen}})

        user_message = [
            {
                "role": "user",
                "content": content,
            },
        ]
        return user_message

    # TODO: debug
    def format_user_message_video(
        self,
        images: Sequence[bytes],
        prompt: str = "Describe clearly what you see in the video",
    ):
        base64_frames = ",".join([self._encode_image(a) for a in images])
        video_content = f"data:video/jpeg;base64,{base64_frames}"

        content = [
            {"type": "video_url", "video_url": {"url": video_content}},
            {"type": "text", "text": prompt},
        ]

        user_message = [
            {
                "role": "user",
                "content": content,
            },
        ]
        return user_message

    def inference_with_api(
        self,
        images: list[bytes],
        prompt: str = "Describe clearly what you see in the video",
        sys_prompt: str = None,
        base_url="http://localhost:8000/v1",
        as_video: bool = False,
    ):
        messages = []
        if sys_prompt:
            messages = [
                {"role": "system", "content": sys_prompt},
            ]

        if as_video:
            user_messages = self.format_user_message_video(images=images, prompt=prompt)
        else:
            user_messages = self.format_user_message_images(
                images=images, prompt=prompt
            )

        messages.extend(user_messages)

        client = openai.OpenAI(base_url=base_url, api_key="sk-no-key-required")

        completion = client.chat.completions.create(
            model="local-model",
            messages=messages,
            response_format={"type": "json_object"},
        )

        return completion.choices[0].message.content


class DspyAnalyzer(Analyzer):
    def __init__(
        self,
        model: str = "openai/Qwen2.5-VL-3B",
        temperature: float = 0.7,
        prompting_mode: str = "basic",
        cache: bool = True,
    ):
        if not model.startswith("openai/"):
            raise ValueError("Only OpenAI compatible endpoints are supported.")

        assert prompting_mode in ["basic", "cot"], f"{prompting_mode} not supported."

        self.lm = load_dspy_model(
            model=model, temperature=temperature, model_type="chat", cache=cache
        )

        # dspy.configure(lm=self.lm)

        if prompting_mode == "cot":
            self.llm = dspy.ChainOfThought(Signature)
        else:
            self.llm = dspy.Predict(Signature)

    def run(self, images: list[bytes]):
        for a in images:
            assert isinstance(a, bytes), f"Expected type 'bytes' but found {type(a)}"

        images = [dspy.Image.from_file(a) for a in images]

        with dspy.context(lm=self.lm):
            response = self.llm(images=images)

        return response.analysis


class Summarizer:
    def __init__(
        self,
        model: str = "openai/Qwen2.5-VL-3B",
        cache: bool = True,
        temperature: float = 0.7,
    ):
        self.lm = load_dspy_model(
            model=model, temperature=temperature, model_type="chat", cache=cache
        )

        self.summarizer = dspy.ChainOfThought(SummarySignature)

    def run(self, texts: list[str]):
        with dspy.context(lm=self.lm):
            response = self.summarizer(image_descriptions=texts)

        return response.summary



