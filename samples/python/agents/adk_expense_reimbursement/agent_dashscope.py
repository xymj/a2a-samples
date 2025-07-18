import json
import logging
import os
import random
import types
from collections.abc import AsyncIterable
from typing import Any, Dict

from a2a.utils import new_agent_text_message
from dashscope import Generation
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.llms.tongyi import Tongyi
import asyncio

logging.basicConfig(level=logging.DEBUG)

# 本地缓存 request_id
request_ids = set()


def create_request_form(date: str = '', amount: str = '', purpose: str = '') -> Dict[str, Any]:
    request_id = 'request_id_' + str(random.randint(1000000, 9999999))
    request_ids.add(request_id)
    return {
        'request_id': request_id,
        'date': date or '<transaction date>',
        'amount': amount or '<transaction dollar amount>',
        'purpose': purpose or '<business justification/purpose of the transaction>',
    }


def return_form(form_request: Dict[str, Any], instructions: str = '') -> str:
    if isinstance(form_request, str):
        form_request = json.loads(form_request)

    print(json.dumps(form_request, indent=2))
    form_dict = {
        'type': 'form',
        'form': {
            'type': 'object',
            'properties': {
                'date': {'type': 'string', 'format': 'date', 'description': 'Date of expense', 'title': 'Date'},
                'amount': {'type': 'string', 'format': 'number', 'description': 'Amount of expense', 'title': 'Amount'},
                'purpose': {'type': 'string', 'description': 'Purpose of expense', 'title': 'Purpose'},
                'request_id': {'type': 'string', 'description': 'Request id', 'title': 'Request ID'},
            },
            'required': list(form_request.keys()),
        },
        'form_data': form_request,
        'instructions': instructions,
    }
    return json.dumps(form_dict)


def reimburse(request_id: str) -> Dict[str, Any]:
    print("reimburse called with request_id:", request_id)
    request_id = json.loads(request_id)["request_id"]
    print("request_id:", request_id)
    if request_id not in request_ids:
        return {'request_id': request_id, 'status': 'Error: Invalid request_id.'}
    return {'request_id': request_id, 'status': 'approved'}


# 封装为 langchain Tool
tools = [
    Tool.from_function(
        func=create_request_form,
        name="create_request_form",
        description="创建报销请求表单"
    ),
    Tool.from_function(
        func=return_form,
        name="return_form",
        description="返回表单结构化数据"
    ),
    Tool.from_function(
        func=reimburse,
        name="reimburse",
        description="报销指定 request_id"
    ),
]

# 初始化 DashScope LLM
llm = Tongyi(
    model_name=Generation.Models.qwen_max,
    temperature=0.7,
    api_key=os.getenv("AI_DASHSCOPE_API_KEY"),
)

# 初始化 Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


# 示例调用
def process_reimbursement(query: str):
    # result = agent.run(query)
    # return result
    result = ''
    for chunk in agent.stream(query):
        result += chunk
        print(f"chunk-> {chunk}")
    return result


class DashScopeReimbursementAgent:
    """An agent that handles reimbursement requests."""

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self):
        self._agent = self._build_agent()
        self._user_id = 'remote_agent'

    def get_processing_message(self) -> str:
        return 'Processing the reimbursement request...'

    def _build_agent(self):
        """Builds the LLM agent for the reimbursement agent."""
        return initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            agent_kwargs={
                "prefix": """
    You are an agent who handles the reimbursement process for employees.

    When you receive a reimbursement request, you should first create a new request form using create_request_form(). Only provide default values if they are provided by the user, otherwise use an empty string as the default value.
      1. 'Date': the date of the transaction.
      2. 'Amount': the dollar amount of the transaction.
      3. 'Business Justification/Purpose': the reason for the reimbursement.

    Once you created the form, you should return the result of calling return_form with the form data from the create_request_form call.

    Once you received the filled-out form back from the user, you should then check the form contains all required information:
      1. 'Date': the date of the transaction.
      2. 'Amount': the value of the amount of the reimbursement being requested.
      3. 'Business Justification/Purpose': the item/object/artifact of the reimbursement.

    If you don't have all of the information, you should reject the request directly by calling the request_form method, providing the missing fields.


    For valid reimbursement requests, you can then use reimburse() to reimburse the employee.
      * In your response, you should include the request_id and the status of the reimbursement request.

    """,
            }
        )

    async def stream(self, query, session_id) -> AsyncIterable[dict[str, Any]]:

        def to_async_iterable(iterator):
            async def async_gen():
                for item in iterator:
                    await asyncio.sleep(0)
                    yield item

            return async_gen()

        async for event in to_async_iterable(self._agent.stream(query)):
            if event.get('output'):
                yield {
                    'is_task_complete': True,
                    'content': event['output'],
                }
            else:
                if event.get('actions') and isinstance(event['actions'], list) and event['actions'][0]:
                    yield {
                        'is_task_complete': False,
                        'updates': event['actions'][0].log,
                    }
                elif event.get('steps') and isinstance(event['steps'], list) and event['steps'][0]:
                    yield {
                        'is_task_complete': False,
                        'updates': event['steps'][0].action.log,
                    }
                else:
                    yield {
                        'is_task_complete': False,
                        'updates': self.get_processing_message(),
                    }
# 用法
if __name__ == "__main__":
    query = "我要报销一笔差旅费用，金额500元，日期2024-06-01，事由为客户拜访"
    print(f"result: {process_reimbursement(query)}")
