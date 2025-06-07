# -*- coding: utf-8 -*-
"""
LiveKit与Dify工作流的集成插件

该模块实现了LiveKit Agents框架与Dify工作流API的无缝集成，
支持流式输出和全部Dify工作流事件处理。

作者: xch
日期: 2025-6-5
"""

import json
import os
import uuid
from dataclasses import dataclass
from typing import Dict, Any, Optional
import aiohttp

from livekit.agents import llm, utils
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    LLMStream,
)
from livekit.agents import APIConnectionError, APIStatusError 


DEFAULT_DIFY_API_BASE = "https://api.dify.ai/v1" 
DEFAULT_DIFY_WORKFLOW_ENDPOINT = "/chat-messages" 

@dataclass
class DifyWorkflowLLMOptions:
    """
    Dify工作流LLM的配置选项
    
    包含连接Dify API所需的各种配置参数，支持从环境变量加载。
    """
    api_key: str  # API密钥，必需参数
    api_base: str = DEFAULT_DIFY_API_BASE  # API基础URL，默认为官方地址
    workflow_api_endpoint: str = DEFAULT_DIFY_WORKFLOW_ENDPOINT  # 工作流端点
    user: Optional[str] = None  # 用户标识
    http_session: Optional[aiohttp.ClientSession] = None  # 可选的HTTP会话

    def __post_init__(self):
        """初始化后处理配置参数，处理环境变量和默认值"""
       
        if self.api_key is None:
            self.api_key = os.environ.get("DIFY_API_KEY")
        
    
        if not self.api_key:
            raise ValueError("Dify API密钥是必需的，并且在选项或DIFY_API_KEY环境变量中未找到。")
        
   
        self.api_base = self.api_base or os.environ.get("DIFY_API_BASE", DEFAULT_DIFY_API_BASE)
        self.user = self.user or os.environ.get("DIFY_USER", "livekit_user")  # 用户ID默认值
        self.workflow_api_endpoint = self.workflow_api_endpoint or DEFAULT_DIFY_WORKFLOW_ENDPOINT

    def get_headers(self) -> Dict[str, str]:
        """生成API请求所需的HTTP头信息"""
        # 注意这里使用Bearer认证
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

class DifyWorkflowLLMStream(llm.LLMStream):
    def __init__(
        self,
        llm_instance: 'DifyWorkflowLLM',
        dify_stream: aiohttp.ClientResponse,
        workflow_inputs: Dict[str, Any],
        user_query: Optional[str],
    ):
        from livekit.agents.llm import ChatContext
        empty_chat_ctx = ChatContext()
        
        from dataclasses import dataclass
        
        @dataclass
        class ConnectionOptions:
            max_retry: int = 3
            retry_wait: float = 1.0
            retry_interval: float = 1.0
        
        conn_opts = ConnectionOptions()
        
        super().__init__(
            llm_instance,
            chat_ctx=empty_chat_ctx,
            tools=[],
            conn_options=conn_opts,
        )
        self._llm = llm_instance
        self._dify_stream = dify_stream
        self._workflow_inputs = workflow_inputs
        self._user_query = user_query
        self._finish_reason = None

    async def _run(self) -> None:
        final_usage_data: Optional[Dict[str, Any]] = None
        final_metadata: Optional[Dict[str, Any]] = None
        workflow_failed_error_message: Optional[str] = None

        try:
            async for line in self._dify_stream.content:
                if not line:
                    continue
                line_str = line.decode('utf-8').strip()
                if not line_str:
                    continue

                if line_str.startswith("data:"):
                    data_str = line_str[len("data:"):].strip()
                    if not data_str:
                        continue
                    
                    try:
                        event_data = json.loads(data_str)
                    except json.JSONDecodeError:
                        self._llm.logger.warning(f"从Dify Workflow流解码JSON失败: {data_str}")
                        continue

                    event_type = event_data.get("event")

                    if event_type == "message":
                        answer = event_data.get("answer")
                        if answer:
                            from livekit.agents.llm import ChoiceDelta
                            chunk = llm.ChatChunk(
                                id=str(uuid.uuid4()),
                                delta=ChoiceDelta(content=str(answer), role='assistant')
                            )
                            await self._event_ch.send(chunk)
                    elif event_type == "workflow_started":
                        self._llm.logger.info(f"Dify Workflow Started: {event_data.get('task_id')}, Run ID: {event_data.get('workflow_run_id')}")

                    elif event_type == "node_started":
                        node_data = event_data.get("data", {})
                        self._llm.logger.debug(f"Dify Node Started: {node_data.get('title')} ({node_data.get('node_id')})")

                    elif event_type == "node_finished":
                        node_data = event_data.get("data", {})
                        node_status = node_data.get("status")
                        self._llm.logger.debug(f"Dify Node Finished: {node_data.get('title')} ({node_data.get('node_id')}), Status: {node_status}")

                    elif event_type == "workflow_finished":
                        self._llm.logger.info(f"Dify Workflow Finished: {event_data.get('task_id')}, Run ID: {event_data.get('workflow_run_id')}")
                        workflow_data = event_data.get("data", {})
                        status = workflow_data.get("status")
                        final_usage_data = workflow_data.get("usage") or (workflow_data.get("metadata", {}) or {}).get("usage")

                        if status == "succeeded":
                            self._finish_reason = "stop"
                            from livekit.agents.llm import ChoiceDelta
                            await self._event_ch.send(llm.ChatChunk(
                                id=str(uuid.uuid4()),
                                delta=ChoiceDelta(content="", role='assistant')
                            ))
                        elif status == "failed":
                            workflow_failed_error_message = workflow_data.get("error", "工作流执行失败但未提供明确错误信息")
                            self._llm.logger.error(f"Dify Workflow Failed: {workflow_failed_error_message}")
                            self._finish_reason = "error"
                            raise Exception(f"Dify Workflow Failed: {workflow_failed_error_message}")
                        else:
                            workflow_failed_error_message = f"工作流以意外状态结束: {status}"
                            self._llm.logger.warning(workflow_failed_error_message)
                            self._finish_reason = "error"
                            raise Exception(f"Dify Workflow Unexpected Status: {status}")

                    elif event_type == "message_end":
                        self._llm.logger.debug(f"Dify Message End: {event_data.get('message_id')}")
                        if not final_usage_data:
                            final_usage_data = (event_data.get("metadata", {}) or {}).get("usage")
                        if not final_metadata:
                            final_metadata = event_data.get("metadata")
                        
                        if self._finish_reason is None:
                             self._finish_reason = "stop"
                             from livekit.agents.llm import ChoiceDelta
                             await self._event_ch.send(llm.ChatChunk(
                                 id=str(uuid.uuid4()),
                                 delta=ChoiceDelta(content="", role='assistant')
                             ))

                    elif event_type == "error":
                        error_msg = event_data.get("message", "Dify流返回了未指定的错误")
                        status_code = event_data.get("status")
                        error_code = event_data.get("code")
                        self._llm.logger.error(f"Dify Stream Error Event: Status {status_code}, Code {error_code}, Message: {error_msg}")
                        raise APIStatusError(f"Dify API流错误: {error_msg} (Code: {error_code}, Status: {status_code})")

            if self._finish_reason is None:
                self._llm.logger.warning("Dify流在没有明确结束事件的情况下终止。")
            
            if workflow_failed_error_message and self._finish_reason is None:
                self._finish_reason = "error"
                raise APIConnectionError(f"Dify工作流失败: {workflow_failed_error_message}")

        except APIStatusError as e:
            self._finish_reason = "error"
            raise e
        except Exception as e:
            self._llm.logger.error(f"处理Dify Workflow流时发生错误: {e}", exc_info=True)
            self._finish_reason = "error"
            raise APIConnectionError(f"处理Dify Workflow流时出错: {e}") from e
        finally:
            if final_usage_data:
                self._parse_usage(final_usage_data)
            await self._dify_stream.release()

    def _parse_usage(self, usage_data: Optional[Dict[str, Any]]) -> None:
        if usage_data:
            self._llm.logger.debug(f"Received Dify usage data: {usage_data}")
        return None

class DifyWorkflowLLM(llm.LLM):
    def __init__(self, options: DifyWorkflowLLMOptions, **kwargs):
        super().__init__()
        self._opts = options
        self._session: Optional[aiohttp.ClientSession] = None
        
        import logging
        self.logger = logging.getLogger("dify_workflow_llm")
        
        self.logger.info(f"DifyWorkflowLLM initialized with options: api_base='{options.api_base}', user='{options.user}'")

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._opts.http_session and not self._opts.http_session.closed:
            self.logger.debug("Using provided http_session") 
            return self._opts.http_session
        
        if self._session and not self._session.closed:
            return self._session
            
        # 直接创建新的ClientSession并禁用SSL验证
        self._session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=False)
        )
        self.logger.debug("Created new aiohttp.ClientSession with SSL verification disabled")
        return self._session

    async def close(self):
        self._session = None

    async def run_workflow(
        self,
        *, 
        workflow_inputs: Dict[str, Any],
        user_query: Optional[str] = None,
        stream: bool = True,
        **kwargs: Any
    ) -> DifyWorkflowLLMStream:
        session = await self._ensure_session()

        payload = {
            "inputs": workflow_inputs,
            "user": self._opts.user,
            "response_mode": "streaming" if stream else "blocking",
        }
        if user_query:
            payload["query"] = user_query

        api_url = f"{self._opts.api_base.rstrip('/')}/{self._opts.workflow_api_endpoint.lstrip('/')}"

        try:
            response = await session.post(
                api_url,
                headers=self._opts.get_headers(),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=600),
                ssl=False  # 禁用SSL验证以解决SSL连接问题
            )
            response.raise_for_status()

        except aiohttp.ClientResponseError as e:
            raise APIStatusError(f"Dify Workflow API 错误: {e.status} {e.message}") from e
        except Exception as e:
            raise APIConnectionError(f"连接 Dify Workflow API 失败: {e}") from e

        return DifyWorkflowLLMStream(
            llm_instance=self,
            dify_stream=response,
            workflow_inputs=workflow_inputs,
            user_query=user_query,
        )

    async def chat(self, chat_ctx: llm.ChatContext, **kwargs) -> DifyWorkflowLLMStream:
        if not chat_ctx.messages:
            raise ValueError("ChatContext必须至少包含一条消息。")

        last_message = chat_ctx.messages[-1]
        user_query = ""
        if last_message.content:
            if isinstance(last_message.content, str):
                user_query = last_message.content
            elif isinstance(last_message.content, list) and last_message.content:
                user_query = str(last_message.content[0])
        
        workflow_inputs = kwargs.pop("workflow_inputs", {})

        return await self.run_workflow(
            workflow_inputs=workflow_inputs, 
            user_query=user_query,
            stream=kwargs.get("stream", True), 
            **kwargs
        )