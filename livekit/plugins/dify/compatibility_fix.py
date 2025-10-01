"""
LiveKit Agents 1.0.23 兼容性修复

此模块提供了针对LiveKit Agents 1.0.23版本的兼容层，解决DifyWorkflowLLM
与LiveKit异步上下文管理器协议和事件系统的不兼容问题。
"""

"""
LiveKit Agents 1.0.23 兼容性修复

此模块提供了针对LiveKit Agents 1.0.23版本的兼容层，解决DifyWorkflowLLM
与LiveKit异步上下文管理器协议和事件系统的不兼容问题。
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, AsyncIterator, List, Union, Callable
from livekit.agents.llm import ChatContext, ChatMessage, LLM, ChatChunk
from livekit.rtc.event_emitter import EventEmitter
from livekit.plugins.dify.llm import DifyWorkflowLLM as OriginalDifyWorkflowLLM
from livekit.plugins.dify.llm import DifyWorkflowLLMOptions, DifyWorkflowLLMStream
import uuid

logger = logging.getLogger("dify_compatibility")


class AsyncContextStream:
    """适配器类：将DifyWorkflowLLMStream转换为支持async with和async for的对象"""
    
    def __init__(self, stream_coroutine):
        self.stream_coroutine = stream_coroutine
        self.stream = None
        
    async def __aenter__(self):
        self.stream = await self.stream_coroutine
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.stream = None
        return False
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if not self.stream:
            raise RuntimeError("Stream was not initialized with 'async with'")
            
        try:
            return await self.stream.__anext__()
        except StopAsyncIteration:
            raise
        except Exception as e:
            logger.error(f"Error in AsyncContextStream.__anext__: {e}")
            raise StopAsyncIteration


class DifyWorkflowLLM(LLM, EventEmitter):
    """LiveKit Agents 1.0.23兼容版本的DifyWorkflowLLM"""
    
    def __init__(self, options: DifyWorkflowLLMOptions, **kwargs: Any):
        EventEmitter.__init__(self)
        self.original_llm = OriginalDifyWorkflowLLM(options,** kwargs)
        self.options = options
        self._kwargs = kwargs
        self._original_chat = self.original_llm.chat
        self.original_llm.chat = self._patched_chat
        logger.info("DifyWorkflowLLM兼容层已初始化")
    
    async def _patched_chat(self, chat_ctx: ChatContext, **kwargs: Any) -> AsyncIterator[ChatChunk]:
        try:
            self.emit("chat_start", chat_ctx=chat_ctx)
            
            # 适配LiveKit 1.0.23的ChatContext结构差异
            if hasattr(chat_ctx, 'items') and not hasattr(chat_ctx, 'messages'):
                messages = []
                for item in chat_ctx.items:
                    if hasattr(item, 'type') and item.type == 'message':
                        messages.append(item)
                chat_ctx.messages = messages
                
                if messages and kwargs.get('prompt') is None:
                    last_message = messages[-1]
                    if hasattr(last_message, 'content'):
                        content = last_message.content
                        if isinstance(content, list) and len(content) > 0 and isinstance(content[0], str):
                            kwargs['prompt'] = content[0]
                            logger.info(f"提取用户查询: {content[0][:30]}...")

            async for chunk in self._original_chat(chat_ctx,** kwargs):
                self.emit("chat_chunk", chunk=chunk)
                yield chunk
                
            self.emit("chat_end", success=True)
        except Exception as e:
            self.emit("chat_error", error=e)
            logger.error(f"_patched_chat发生错误: {e}")
            
            async def error_generator():
                yield ChatChunk(
                    id=str(uuid.uuid4()),
                    delta=ChatChunk.ChoiceDelta(content="抱歉，处理请求时发生错误", role="assistant")
                )
            async for chunk in error_generator():
                yield chunk
    
    def chat(self, chat_ctx: ChatContext, **kwargs: Any) -> AsyncContextStream:
        """返回支持异步上下文管理器的流对象"""
        chat_coroutine = self.original_llm.chat(chat_ctx,** kwargs)
        return AsyncContextStream(chat_coroutine)

    async def run_workflow(self, 
                          workflow_inputs: Optional[Dict[str, Any]] = None, 
                          user_query: Optional[str] = None, 
                          stream: bool = True, 
                          conversation_id: Optional[str] = None, 
                          **kwargs: Any):
        """直接调用原始run_workflow方法"""
        return await self.original_llm.run_workflow(
            workflow_inputs=workflow_inputs,
            user_query=user_query,
            stream=stream,
            conversation_id=conversation_id,** kwargs
        )

    async def close(self) -> None:
        """关闭资源"""
        await self.original_llm.close()
