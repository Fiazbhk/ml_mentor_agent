import logging
import os
import json
import asyncio
import traceback
import urllib.parse
from dotenv import load_dotenv

# LangChain / Coral imports
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool

# Import domain-specific solver
from ml_solver import MLSolver

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Example tool wrapper to expose solver to LLM as a callable tool
async def ml_solver_tool(query: str) -> str:
    solver = MLSolver()
    return await solver.solve_problem(query)


def get_tools_description(tools):
    return "\n".join(f"Tool: {t.name}, Schema: {json.dumps(getattr(t, 'args_schema', {}))}" for t in tools)


async def create_agent(coral_tools, agent_tools):
    combined_tools = coral_tools + agent_tools

    # System prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a friendly, patient Machine Learning Mentor. "
            "When given a question, decide whether to: explain concepts, debug code, suggest experiments, or propose a lesson plan. "
            "If the question requires running code or tools, call the appropriate tool. "
        )),
        ("placeholder", "{agent_scratchpad}")
    ])

    model = init_chat_model(
        model=os.getenv("MODEL_NAME", "gpt-4.1"),
        model_provider=os.getenv("MODEL_PROVIDER", "openai"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=float(os.getenv("MODEL_TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("MODEL_MAX_TOKENS", "8000"))
    )

    agent = create_tool_calling_agent(model, combined_tools, prompt)
    return AgentExecutor(agent=agent, tools=combined_tools, verbose=True)


async def main():
    load_dotenv()

    base_url = os.getenv("CORAL_SSE_URL")
    agent_id = os.getenv("CORAL_AGENT_ID")

    if not base_url or not agent_id:
        logger.warning("CORAL_SSE_URL or CORAL_AGENT_ID not set. Running in local mode.")

    coral_params = {
        "agentId": agent_id or "ml-mentor-agent",
        "agentDescription": "Machine Learning Mentor Agent"
    }

    client = MultiServerMCPClient({
        "coral": {
            "transport": "sse",
            "url": f"{base_url}?{urllib.parse.urlencode(coral_params)}" if base_url else "",
            "timeout": int(os.getenv("TIMEOUT_MS", 300))
        }
    })

    # Try to get coral tools; if fails, continue with only custom tools
    coral_tools = []
    try:
        if base_url:
            coral_tools = await client.get_tools(server_name="coral")
    except Exception as e:
        logger.error("Failed to fetch Coral tools: %s", e)

    # Define our custom tool
    agent_tools = [
        Tool(
            name="ml_solver_tool",
            func=ml_solver_tool,
            description="Solve ML questions: explain concepts, debug, propose lessons"
        )
    ]

    agent_executor = await create_agent(coral_tools, agent_tools)

    # Demo queries (in real use, Coral will send input)
    sample_queries = [
        "Explain bias-variance tradeoff with examples",
        "Debug: my training loss increases — here's snippet: for epoch in range(5): ...",
        "Design a one-week lesson plan to teach Transformers to a graduate student"
    ]

    for q in sample_queries:
        logger.info("Invoking agent on query: %s", q)
        await agent_executor.ainvoke({"agent_scratchpad": q})
        await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(main())
