from typing import AsyncIterator, List, Dict, Any, Callable, Literal
from loguru import logger

from .agent_interface import AgentInterface
from ..output_types import SentenceOutput, DisplayText
from ..stateless_llm.stateless_llm_interface import StatelessLLMInterface
from ...chat_history_manager import get_history
from ..transformers import (
    sentence_divider,
    actions_extractor,
    tts_filter,
    display_processor,
)
from ...config_manager import TTSPreprocessorConfig
from ..input_types import BatchInput, TextSource, ImageSource
from prompts import prompt_loader
from ...rag import RAGSystem


class RAGAgent(AgentInterface):
    """
    Agent with RAG capabilities for enhanced responses using knowledge base.
    Implements text-based responses with sentence processing pipeline.
    """

    _system: str = """You are an error message repeater. 
        Your job is repeating this error message: 
        'No system prompt set. Please set a system prompt'. 
        Don't say anything else.
        """

    def __init__(
        self,
        llm: StatelessLLMInterface,
        system: str,
        live2d_model,
        rag_config: Dict[str, Any],
        tts_preprocessor_config: TTSPreprocessorConfig = None,
        faster_first_response: bool = True,
        segment_method: str = "pysbd",
        interrupt_method: Literal["system", "user"] = "user",
    ):
        """
        Initialize the agent with LLM, system prompt, RAG module and configuration

        Args:
            llm: `StatelessLLMInterface` - The LLM to use
            system: `str` - System prompt
            live2d_model: `Live2dModel` - Model for expression extraction
            rag_config: `Dict[str, Any]` - Configuration for RAG module
            tts_preprocessor_config: `TTSPreprocessorConfig` - Configuration for TTS preprocessing
            faster_first_response: `bool` - Whether to enable faster first response
            segment_method: `str` - Method for sentence segmentation
            interrupt_method: `Literal["system", "user"]` - Methods for writing interruptions signal
        """
        super().__init__()
        self._memory = []
        self._live2d_model = live2d_model
        self._tts_preprocessor_config = tts_preprocessor_config
        self._faster_first_response = faster_first_response
        self._segment_method = segment_method
        self.interrupt_method = interrupt_method
        self._interrupt_handled = False
        self._set_llm(llm)
        self.set_system(system)
        
        # Initialize RAG module
        self._rag = RAGSystem(rag_config)
        self._rag.initialize()
        logger.info("RAGAgent initialized with RAG module.")

    def _set_llm(self, llm: StatelessLLMInterface):
        """Set the LLM to be used for chat completion."""
        self._llm = llm
        self.chat = self._chat_function_factory(llm.chat_completion)

    def set_system(self, system: str):
        """Set the system prompt"""
        logger.debug(f"RAG Agent: Setting system prompt: '''{system}'''")

        if self.interrupt_method == "user":
            system = f"{system}\n\nIf you received `[interrupted by user]` signal, you were interrupted."

        self._system = system

    def _add_message(
        self,
        message: str | List[Dict[str, Any]],
        role: str,
        display_text: DisplayText | None = None,
    ):
        """Add a message to the memory"""
        if isinstance(message, list):
            text_content = ""
            for item in message:
                if item.get("type") == "text":
                    text_content += item["text"]
        else:
            text_content = message

        message_data = {
            "role": role,
            "content": text_content,
        }

        if display_text:
            if display_text.name:
                message_data["name"] = display_text.name
            if display_text.avatar:
                message_data["avatar"] = display_text.avatar

        self._memory.append(message_data)

    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        """Load the memory from chat history"""
        messages = get_history(conf_uid, history_uid)

        self._memory = []
        self._memory.append(
            {
                "role": "system",
                "content": self._system,
            }
        )

        for msg in messages:
            self._memory.append(
                {
                    "role": "user" if msg["role"] == "human" else "assistant",
                    "content": msg["content"],
                }
            )

    def handle_interrupt(self, heard_response: str) -> None:
        """Handle an interruption by the user."""
        if self._interrupt_handled:
            return

        self._interrupt_handled = True

        if self._memory and self._memory[-1]["role"] == "assistant":
            self._memory[-1]["content"] = heard_response + "..."
        else:
            if heard_response:
                self._memory.append(
                    {
                        "role": "assistant",
                        "content": heard_response + "...",
                    }
                )
        self._memory.append(
            {
                "role": "system" if self.interrupt_method == "system" else "user",
                "content": "[Interrupted by user]",
            }
        )

    def _to_text_prompt(self, input_data: BatchInput) -> str:
        """Format BatchInput into a prompt string"""
        message_parts = []

        for text_data in input_data.texts:
            if text_data.source == TextSource.INPUT:
                message_parts.append(text_data.content)
            elif text_data.source == TextSource.CLIPBOARD:
                message_parts.append(f"[Clipboard content: {text_data.content}]")

        if input_data.images:
            message_parts.append("\nImages in this message:")
            for i, img_data in enumerate(input_data.images, 1):
                source_desc = {
                    ImageSource.CAMERA: "captured from camera",
                    ImageSource.SCREEN: "screenshot",
                    ImageSource.CLIPBOARD: "from clipboard",
                    ImageSource.UPLOAD: "uploaded",
                }[img_data.source]
                message_parts.append(f"- Image {i} ({source_desc})")

        return "\n".join(message_parts)

    async def _to_messages(self, input_data: BatchInput) -> List[Dict[str, Any]]:
        """Prepare messages list with RAG context"""
        messages = self._memory.copy()
        
        # Get user query
        user_query = self._to_text_prompt(input_data)
        
        # Get conversation history for context
        history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self._memory
            if msg["role"] in ["user", "assistant"]
        ]
        
        # Handle empty input (seek proactively case)
        if not user_query.strip():
            # Use last few messages as context for RAG query
            context_for_query = " ".join([
                msg["content"] for msg in history[-3:]  # 使用最近的3条消息
                if msg["content"] and not msg["content"].startswith("[")  # 排除系统消息
            ])
            
            if context_for_query:
                retrieved_context = await self._rag.query(
                    f"Based on this conversation context, what would be relevant to discuss next: {context_for_query}",
                    conversation_history=history
                )
            else:
                # 如果没有最近的对话历史，使用一个通用的查询
                retrieved_context = await self._rag.query(
                    "What would be an interesting topic to discuss?",
                    conversation_history=None
                )
        else:
            # Normal case with user input
            retrieved_context = await self._rag.query(user_query, conversation_history=history)
        
        # Create context message
        if retrieved_context:
            context_message = {
                "role": "system",
                "content": f"Here is some relevant information that might help you answer:\n{retrieved_context}\n\nPlease use this information along with your knowledge to answer the question naturally, as if you're having a casual conversation."
            }
            messages.append(context_message)

        # Add user message
        if input_data.images:
            content = []
            content.append({"type": "text", "text": user_query})
            for img_data in input_data.images:
                content.append({"type": "image_url", "image_url": img_data.url})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user_query})

        return messages

    def _chat_function_factory(
        self, chat_func: Callable[[List[Dict[str, Any]], str], AsyncIterator[str]]
    ) -> Callable[..., AsyncIterator[SentenceOutput]]:
        """Create the chat pipeline with transformers"""

        @tts_filter(self._tts_preprocessor_config)
        @display_processor()
        @actions_extractor(self._live2d_model)
        @sentence_divider(
            faster_first_response=self._faster_first_response,
            segment_method=self._segment_method,
            valid_tags=["think"],
        )
        async def chat_with_memory(input_data: BatchInput) -> AsyncIterator[str]:
            """Chat implementation with memory and RAG"""
            messages = await self._to_messages(input_data)

            token_stream = chat_func(messages, self._system)
            complete_response = ""

            async for token in token_stream:
                yield token
                complete_response += token

            self._add_message(complete_response, "assistant")

        return chat_with_memory

    async def chat(self, input_data: BatchInput) -> AsyncIterator[SentenceOutput]:
        """Placeholder chat method that will be replaced at runtime"""
        return self.chat(input_data)

    def reset_interrupt(self) -> None:
        """Reset the interrupt handled flag"""
        self._interrupt_handled = False

    def start_group_conversation(
        self, human_name: str, ai_participants: List[str]
    ) -> None:
        """Start a group conversation"""
        other_ais = ", ".join(name for name in ai_participants)
        group_context = prompt_loader.load_util("group_conversation_prompt").format(
            human_name=human_name, other_ais=other_ais
        )
        self._memory.append({"role": "user", "content": group_context})
        logger.debug(f"Added group conversation context: '''{group_context}'''") 