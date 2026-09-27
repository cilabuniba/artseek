import json
import random
import re
import string
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
from pathlib import Path

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, ToolCall, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, run_in_executor
from langchain_core.runnables.utils import Output
from langchain_core.tools import BaseTool
from PIL import Image
from transformers import (
    AutoProcessor,
    ProcessorMixin,
    Qwen2_5_VLForConditionalGeneration,
)


class Qwen2_5_VLChatModel(BaseChatModel):
    """A langchain chat model wrapping Qwen2.5-VL run with transformers `generate`.

    Messages may contain `{"type": "image"}` placeholders; the matching PIL
    images are passed separately through the `images` keyword of `invoke`.
    Tool calls are parsed from Qwen's Hermes-style `<tool_call>` blocks.
    """

    processor: ProcessorMixin
    """The processor used for tokenization and image processing."""
    mllm: Qwen2_5_VLForConditionalGeneration
    """The underlying model used for generation."""
    lc_type2hf_roles_map: dict[str, str] = {
        "ai": "assistant",
        "human": "user",
        "system": "system",
        "tool": "tool",
    }

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs: Any,
    ) -> "Qwen2_5_VLChatModel":
        """Load a pre-trained model from the Hugging Face Hub.

        Args:
            model_name_or_path: The model name or path to load from the Hugging Face Hub.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            A new instance of the model.
        """
        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            max_pixels=1280 * 28 * 28,
            local_files_only=True,  # revision="refs/pr/24"
        )
        processor.tokenizer.padding_side = "left"
        with open(
            Path(__file__).resolve().parent / "qwen2_5_vl_chat_template.txt", "r"
        ) as f:
            processor.chat_template = f.read()
        mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=True,
            # torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            # device_map="auto",
            **kwargs,
        )
        return cls(processor=processor, mllm=mllm)

    def _try_parse_tool_calls(self, content: str):
        """Try parse the tool calls."""

        def generate_random_string(length=24):
            characters = (
                string.ascii_letters + string.digits
            )  # Uppercase, lowercase, and digits
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
            additional_kwargs={},  # Used to add additional payload (e.g., function calling request)
            response_metadata={},  # Used for response metadata
            # usage_metadata={},  # Used for additional usage metadata (e.g. no. of tokens used)
        )

    def _preprocess(
        self,
        messages,
        images,
        tools=None,
    ):
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

        # Process text and images
        text = self.processor.apply_chat_template(
            hf_messages, tools=tools, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text,
            images=flattened_images if flattened_images else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.mllm.device)

        return inputs

    def _infer(self, inputs, max_new_tokens):
        """Encapsulate the inference logic."""
        generated_ids = self.mllm.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
        )[0]
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

        This can be a call to an API, a call to a local model, or any other
        implementation that generates a response to the input prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        # Unpack kwargs
        images = kwargs.get("images", [])
        max_new_tokens = kwargs.get("max_new_tokens", 128)
        tools = kwargs.get("tools", None)

        # Preprocess
        inputs = self._preprocess(messages, images, tools)

        # Inference
        output_text = self._infer(inputs, max_new_tokens)

        # Parse content and tools
        result_message = self._try_parse_tool_calls(output_text)
        generation = ChatGeneration(message=result_message)
        return ChatResult(generations=[generation])

    def hf_generate(
        self,
        messages: List[List[BaseMessage]],
        images: List[Image.Image],
        **kwargs,
    ) -> List[str]:
        """Generate text from a batch of messages, using the underlying model.

        Args:
            messages (List[List[BaseMessage]]): A list of conversations.
            images (List[Image.Image]): Images to be processed.

        Returns:
            List[str]: A list of generated responses.
        """
        max_new_tokens = kwargs.get("max_new_tokens", 128)

        inputs = self._preprocess(messages, images)

        # Inference
        generated_ids = self.mllm.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],  # noqa: UP006
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools to the model.

        Please, refer to https://qwen.readthedocs.io/en/latest/framework/function_call.html#hugging-face-transformers
        for more information on how to bind tools to a HF Transformers model.
        """
        return super().bind(tools=tools, **kwargs)

    @property
    def device(self):
        """The device the underlying model runs on (used to place auxiliary
        modules like LICN on the same device)."""
        return self.mllm.device

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "multimodal-to-text"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return self.mllm.config.to_dict()
