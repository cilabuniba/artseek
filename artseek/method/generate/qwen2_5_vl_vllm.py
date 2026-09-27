import json
import random
import re
import string
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
)

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from transformers import AutoProcessor, ProcessorMixin
import torch
from vllm import LLM, SamplingParams


class Qwen2_5_VLVLLMChatModel(BaseChatModel):
    """A langchain chat model wrapping Qwen2.5-VL served through vLLM.

    Mirrors the interface of ``Qwen2_5_VLChatModel`` (the transformers-backed
    implementation) so both can be swapped in ``Qwen2_5_VLRAGModel`` via the
    ``engine`` parameter, but talks to a local ``vllm.LLM`` instance instead
    of calling ``.generate()`` on a transformers model directly.
    """

    processor: ProcessorMixin
    """The processor used for tokenization and image processing."""
    llm: Any
    """The underlying vllm.LLM engine used for generation."""
    lc_type2hf_roles_map: dict[str, str] = {
        "ai": "assistant",
        "human": "user",
        "system": "system",
        "tool": "tool",
    }

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs: Any,
    ) -> "Qwen2_5_VLVLLMChatModel":
        """Load a pre-trained model from the Hugging Face Hub via vLLM.

        Args:
            pretrained_model_name_or_path: The model name or path to load from the Hugging Face Hub.
            **kwargs: Additional keyword arguments forwarded to ``vllm.LLM(...)``
                (e.g. ``dtype``, ``quantization``, ``max_model_len``,
                ``gpu_memory_utilization``, ``limit_mm_per_prompt``,
                ``mm_processor_kwargs``, ``tensor_parallel_size``).

        Returns:
            A new instance of the model.
        """
        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            max_pixels=1280 * 28 * 28,
            local_files_only=True,
        )
        processor.tokenizer.padding_side = "left"
        with open(
            Path(__file__).resolve().parent / "qwen2_5_vl_chat_template.txt", "r"
        ) as f:
            processor.chat_template = f.read()

        llm = LLM(
            model=str(pretrained_model_name_or_path),
            trust_remote_code=True,
            **kwargs,
        )
        return cls(processor=processor, llm=llm)

    def _try_parse_tool_calls(self, content: str):
        """Try parse the tool calls."""

        def generate_random_string(length=24):
            characters = string.ascii_letters + string.digits
            return "".join(random.choices(characters, k=length))

        tool_calls = []
        offset = 0
        for i, m in enumerate(
            re.finditer(r"<tool_call>\n(.+)?\n</tool_call>", content)
        ):
            if i == 0:
                offset = m.start()
            try:
                func = json.loads(m.group(1))
                if isinstance(func["arguments"], str):
                    func["arguments"] = json.loads(func["arguments"])
                tool_calls.append(
                    ToolCall(
                        id=f"call_{generate_random_string(24)}",
                        name=func["name"],
                        args=func["arguments"],
                    )
                )
            except json.JSONDecodeError as e:
                print(
                    f"Failed to parse tool calls: the content is {m.group(1)} and {e}"
                )
                pass
        if tool_calls:
            if offset > 0 and content[:offset].strip():
                c = content[:offset]
            else:
                c = ""
            return AIMessage(
                content=c,
                additional_kwargs={},
                response_metadata={},
                tool_calls=tool_calls,
            )
        return AIMessage(
            content=re.sub(r"<\|im_end\|>$", "", content),
            additional_kwargs={},
            response_metadata={},
        )

    def _preprocess(
        self,
        messages,
        images,
        tools=None,
    ) -> tuple[str, list]:
        hf_messages = []

        # Convert messages or batch of messages
        if isinstance(messages[0], list):
            for message_group in messages:
                hf_message_group = []
                for message in message_group:
                    hf_role = self.lc_type2hf_roles_map[message.type]
                    hf_content = message.content
                    hf_message_group.append({"role": hf_role, "content": hf_content})
                hf_messages.append(hf_message_group)
        else:
            for message in messages:
                hf_role = self.lc_type2hf_roles_map[message.type]
                hf_content = message.content
                hf_message = {
                    "role": hf_role,
                    "content": hf_content,
                }

                if tool_calls := getattr(message, "tool_calls", None):
                    hf_tool_calls = []
                    for tool_call in tool_calls:
                        hf_tool_call = {
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": tool_call["args"],
                            },
                        }
                        hf_tool_calls.append(hf_tool_call)
                    hf_message["tool_calls"] = hf_tool_calls

                if hf_role == "tool":
                    hf_message["name"] = message.name
                hf_messages.append(hf_message)

        # Flatten images if required
        flattened_images = images
        if images and isinstance(images[0], list):
            flattened_images = [
                image for image_group in images for image in image_group
            ]

        # Render the prompt text (image placeholders included); vLLM handles
        # the actual image preprocessing internally given the raw PIL images.
        text = self.processor.apply_chat_template(
            hf_messages, tools=tools, tokenize=False, add_generation_prompt=True
        )

        return text, flattened_images

    def _infer(self, text: str, images: list, max_new_tokens: int) -> str:
        """Encapsulate the inference logic."""
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        request: dict = {"prompt": text}
        if images:
            request["multi_modal_data"] = {"image": images}
        outputs = self.llm.generate([request], sampling_params, use_tqdm=False)
        output_text = outputs[0].outputs[0].text
        output_text = output_text.replace("\\n", "\n")
        return output_text

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override the _generate method to implement the chat model logic.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
            run_manager: A run manager with callbacks for the LLM.
        """
        images = kwargs.get("images", [])
        max_new_tokens = kwargs.get("max_new_tokens", 128)
        tools = kwargs.get("tools", None)

        text, flattened_images = self._preprocess(messages, images, tools)
        output_text = self._infer(text, flattened_images, max_new_tokens)

        result_message = self._try_parse_tool_calls(output_text)
        generation = ChatGeneration(message=result_message)
        return ChatResult(generations=[generation])

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],  # noqa: UP006
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools to the model.

        Please, refer to https://qwen.readthedocs.io/en/latest/framework/function_call.html#hugging-face-transformers
        for more information on how to bind tools to a HF-style chat model
        (the same Hermes-style prompt format is used here).
        """
        return super().bind(tools=tools, **kwargs)

    @property
    def device(self):
        """The device auxiliary modules (e.g. LICN) should be placed on.

        vLLM manages the Qwen model's own device placement internally, so
        this just points at the local accelerator rather than introspecting
        the vLLM engine.
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "multimodal-to-text-vllm"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {"engine": "vllm"}
