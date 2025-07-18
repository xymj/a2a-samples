import logging
import os

import click
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from agent_executor import QnAAgentExecutor
# OpenTelemetry Python 库的核心作用是提供 分布式追踪、指标收集、日志记录 的标准化工具，并支持与多种后端（如 Jaeger、Prometheus）集成，帮助开发者观测、调试和优化分布式系统的性能
# OpenTelemetry 相关模块用于分布式追踪：
#   trace：管理追踪上下文。
from opentelemetry import trace
#   OTLPSpanExporter：将追踪数据导出到 OTLP 兼容的后端（如 Jaeger）。
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
#   StarletteInstrumentor：为 Starlette 应用自动注入追踪逻辑。
from opentelemetry.instrumentation.starlette import StarletteInstrumentor
from opentelemetry.sdk.resources import Resource
#   TracerProvider：生成追踪器（Tracer）。
from opentelemetry.sdk.trace import TracerProvider
# 添加 BatchSpanProcessor 实现异步批量导出追踪数据。
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from agent_executor_dashscope import QnADashScopeAgentExecutor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10020)
def main(host: str, port: int):
    """A2A Telemetry Sample GRPC Server."""
    # if not os.getenv('GOOGLE_API_KEY'):
    #     raise ValueError('GOOGLE_API_KEY is not set.')

    skill = AgentSkill(
        id='question_answer',
        name='Q&A Agent',
        description='A helpful assistant agent that can answer questions.',
        tags=['Question-Answer'],
        examples=[
            'Who is leading 2025 F1 Standings?',
            'Where can i find an active volcano?',
        ],
    )

    # agent_executor = QnAAgentExecutor()
    agent_executor = QnADashScopeAgentExecutor()
    agent_card = AgentCard(
        name='Q&A Agent',
        description='A helpful assistant agent that can answer questions.',
        url=f'http://{host}:{port}/',
        version='1.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, task_store=InMemoryTaskStore()
    )

    logger.debug('Telemetry Configuration')
    # Set the service name for query back in Jaeger and Grafana
    resource = Resource(
        attributes={
            'service.name': 'a2a-telemetry-sample',
            'service.version': '1.0',
        }
    )
    # Create a TracerProvider and register it
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)
    tracer_provider = trace.get_tracer_provider()

    # Create and configure Jaeger exporter, UDP transport.
    jaeger_exporter = OTLPSpanExporter(
        endpoint='http://localhost:4317', insecure=True
    )
    tracer_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))

    server = A2AStarletteApplication(agent_card, request_handler)
    starlette_app = server.build()
    # Instrument the starlette app for tracing
    StarletteInstrumentor().instrument_app(starlette_app)
    uvicorn.run(starlette_app, host=host, port=port)


if __name__ == '__main__':
    main()
