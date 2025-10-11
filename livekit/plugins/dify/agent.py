from livekit.agents import Agent, AgentSession, JobContext, RunContext
from typing import Optional
import logging
from .workflow import DifyWorkflowPlugin
from .config import DifyConfig

logger = logging.getLogger(__name__)

class DifyAgent(Agent):
    """集成Dify工作流的LiveKit Agent"""
    
    def __init__(self, dify_config: DifyConfig):
        super().__init__()
        self.dify_plugin = DifyWorkflowPlugin(dify_config)
        self.setup_event_handlers()
    
    def setup_event_handlers(self):
        """设置事件处理器"""
        self.dify_plugin.on_event('workflow_start', self._on_workflow_start)
        self.dify_plugin.on_event('workflow_progress', self._on_workflow_progress)
        self.dify_plugin.on_event('workflow_complete', self._on_workflow_complete)
        self.dify_plugin.on_event('error', self._on_error)
    
    async def _on_workflow_start(self, data):
        """工作流开始事件"""
        logger.info(f"开始处理工作流: {data['query']}")
    
    async def _on_workflow_progress(self, data):
        """工作流进度事件"""
        logger.debug(f"工作流进度: {data}")
    
    async def _on_workflow_complete(self, data):
        """工作流完成事件"""
        logger.info(f"工作流完成: {data['query']}")
    
    async def _on_error(self, data):
        """错误事件"""
        logger.error(f"工作流错误: {data['error']}")
    
    async def start(self, ctx: RunContext):
        """启动Agent"""
        await self.dify_plugin.start()
        logger.info("Dify Agent已启动")
    
    async def stop(self):
        """停止Agent"""
        await self.dify_plugin.stop()
        logger.info("Dify Agent已停止")
    
    async def handle_user_message(self, session: AgentSession, message: str):
        """处理用户消息"""
        try:
            # 获取用户ID
            user_id = session.participant.identity or "anonymous"
            
            # 处理消息
            async for chunk in self.dify_plugin.process_message(message, user_id):
                if chunk.get('answer'):
                    # 发送回复到用户
                    await session.say(chunk['answer'])
                
                # 处理其他类型的响应
                if chunk.get('metadata'):
                    logger.info(f"工作流元数据: {chunk['metadata']}")
                    
        except Exception as e:
            logger.error(f"处理用户消息错误: {e}")
            await session.say("抱歉，处理您的请求时出现错误，请稍后再试。")
