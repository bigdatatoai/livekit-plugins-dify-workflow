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
from livekit.agents.llm import ChatContext, ChatMessage, LLM
from livekit.rtc.event_emitter import EventEmitter
from livekit.plugins.dify.llm import DifyWorkflowLLM as OriginalDifyWorkflowLLM
from livekit.plugins.dify.llm import DifyWorkflowLLMOptions, DifyWorkflowLLMStream
import uuid

logger = logging.getLogger("dify_compatibility")

class AsyncContextStream:
    """
    适配器类：将DifyWorkflowLLMStream转换为支持async with和async for的对象
    """
    
    def __init__(self, stream_coroutine):
        self.stream_coroutine = stream_coroutine
        self.stream = None
        
    async def __aenter__(self):
        # 实际获取底层流
        self.stream = await self.stream_coroutine
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # 清理资源
        self.stream = None
        return False
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if not self.stream:
            raise RuntimeError("Stream was not initialized with 'async with'")
            
        try:
            # 直接代理到原始流的__anext__
            return await self.stream.__anext__()
        except StopAsyncIteration:
            raise
        except Exception as e:
            logger.error(f"Error in AsyncContextStream.__anext__: {e}")
            raise StopAsyncIteration


class DifyWorkflowLLM(LLM, EventEmitter):
    """
    LiveKit Agents 1.0.23兼容版本的DifyWorkflowLLM
    完全实现LLM接口，并支持EventEmitter事件系统
    """
    
    def __init__(self, options: DifyWorkflowLLMOptions, **kwargs):
        """初始化兼容层，创建内部原始DifyWorkflowLLM实例"""
        # 先初始化EventEmitter出于多重继承的原因
        EventEmitter.__init__(self)
        
        # 初始化原始LLM
        self.original_llm = OriginalDifyWorkflowLLM(options, **kwargs)
        
        # 保存配置
        self.options = options
        self._kwargs = kwargs
        
        # 替换原始llm的chat方法
        self._original_chat = self.original_llm.chat
        self.original_llm.chat = self._patched_chat
        
        logger.info("DifyWorkflowLLM兼容层已初始化")
    
    async def _patched_chat(self, chat_ctx: ChatContext, **kwargs):
        """
        适配LiveKit Agents 1.0.23的ChatContext
        """
        try:
            # 原始DifyWorkflowLLM期望ChatContext有messages属性，而LiveKit 1.0.23使用的是items
            # 如果items存在，我们创建一个临时的messages属性
            if hasattr(chat_ctx, 'items') and not hasattr(chat_ctx, 'messages'):
                messages = []
                for item in chat_ctx.items:
                    # 只处理类型为message的item
                    if hasattr(item, 'type') and item.type == 'message':
                        messages.append(item)
                
                # 动态添加messages属性
                chat_ctx.messages = messages
                
                # 提取最后一条消息作为用户查询
                if messages and kwargs.get('prompt') is None:
                    last_message = messages[-1]
                    if hasattr(last_message, 'content'):
                        content = last_message.content
                        if isinstance(content, list) and len(content) > 0:
                            if isinstance(content[0], str):
                                # 从content列表中提取文本
                                user_query = content[0]
                                kwargs['prompt'] = user_query
                                logger.info(f"从ChatContext提取用户查询: {user_query[:30]}...")
                
            # 调用原始方法
            return await self._original_chat(chat_ctx, **kwargs)
        except Exception as e:
            logger.error(f"_patched_chat发生错误: {e}")
            # 创建一个空的异步生成器作为fallback
            async def empty_generator():
                yield "抱歉，处理您的请求时发生了错误。"
                
            return empty_generator()
    
    def chat(self, chat_ctx: ChatContext, **kwargs) -> AsyncContextStream:
        """
        兼容LiveKit Agents 1.0.23的chat方法
        返回一个支持异步上下文管理器协议的对象
        """
        # 获取chat协程但不执行它
        chat_coroutine = self.original_llm.chat(chat_ctx, **kwargs)

        
        # 包装为支持异步上下文管理器协议的对象
        return AsyncContextStream(chat_coroutine)

    async def run_workflow(self, workflow_inputs=None, user_query=None, stream=True, conversation_id=None, **kwargs):
        """直接调用原始DifyWorkflowLLM的run_workflow方法"""
        return await self.original_llm.run_workflow(
            workflow_inputs=workflow_inputs,
            user_query=user_query,
            stream=stream,
            conversation_id=conversation_id,
            **kwargs
        )