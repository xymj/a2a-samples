import logging
from collections.abc import Generator

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, TextPart, UnsupportedOperationError
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError
from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search_tool
from google.genai import types

import os
from dashscope import Generation

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QnADashScopeAgentExecutor(AgentExecutor):
    """Agent executor that uses the ADK to answer questions."""

    def __init__(self):
        self.name = None

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise ServerError(error=UnsupportedOperationError())

    async def stream_llm(self, content: str) -> Generator[tuple[bool, str]]:
        messages = [
            {'role': 'system',
             'content': 'A helpful assistant agent that can answer questions.'},
            {'role': 'user', 'content': content}
        ]
        responses = Generation.call(
            api_key=os.environ['AI_DASHSCOPE_API_KEY'], model=Generation.Models.qwen_plus, messages=messages,
            result_format="message", stream=True, incremental_output=True, enable_search=True)
        result = ''
        for chunk in responses:
            choice = chunk.output.choices[0]
            is_final_response = choice.finish_reason == "stop"
            mc = choice.message.content
            result += mc
            yield (is_final_response, result) if is_final_response else (is_final_response, mc)

    async def execute(
            self,
            context: RequestContext,
            event_queue: EventQueue,
    ) -> None:
        logger.debug(f'Executing agent {self.name}')

        query = context.get_user_input()

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        if not context.current_task:
            await updater.submit()

        await updater.start_work()

        async for event in self.stream_llm(query):
            logger.debug(f'Event from ADK {event}')
            if event[0]:
                text_parts = [
                    TextPart(text=event[1]),
                ]
                await updater.add_artifact(
                    text_parts,
                    name='result',
                )
                await updater.complete()
                break
            await updater.update_status(
                TaskState.working, message=new_agent_text_message(f'Working... -> {event[1]}')
            )
        else:
            logger.debug('Agent failed to complete')
            await updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(
                    'Failed to generate a response.'
                ),
            )
