from typing import Literal
from typing_extensions import TypedDict
from typing import Dict, List, Literal, cast
from langgraph.graph import StateGraph, END, START
from tools import TOOLS
from state import State, InputState
from configuration import Configuration
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
from utils import load_chat_model
import asyncio


async def call_model(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering the angent"""
    configuration = Configuration.from_runnable_config(config)
    
    model = load_chat_model(configuration.model).bind_tools(TOOLS)
    
    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = configuration.system_prompt
    
    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages], config
        ),
    )
    
    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.
    
    Args:
        state(State): The current state of the conversation.
        
    Returns:
        str: The name of the next node to call("tools" or "__end__").
    """
    last_massage = state.messages[-1]
    if not isinstance(last_massage, AIMessage):
      raise ValueError(
          f"Expected AIMessage in output edges, but got {type(last_massage).__name__}"
      )  
    # If there is no tool call, then we finish
    if not last_massage.tool_calls:
        return "__end__"
    # Otherwise, we execute the requested actions
    return "tools"


# Define a new graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)
builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint as `call_model`
builder.add_edge(START, "call_model")
# Add a conditional edge to determine the next step after 'call_model'
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # base on the output from the route_model_output
    route_model_output,
)

# Add a normal edge from 'tools' to 'call_model'
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "call_model")

# Compile the builder into an executable graph
# You can customize this by adding interrupt points for state updates
graph = builder.compile(
    interrupt_before=[],  # Add node names here to update state before they're called
    interrupt_after=[],  # Add node names here to update state after they're called
)
graph.name = "Linear Solver Agent"  # This customizes the name in LangSmith

async def main():
   async for chunk in graph.astream(
        {"messages": [("Solve the following system of linear equations using the Jacobi iterative method: \n"
                            "8x - 3y + 2z = 20\n"
                            "4x + 11y - z = 33\n"
                            "6x + 3y + 12z = 36\n"
                            "with an error tolerance of 1e-3 and a maximum iteration limit of 100.")]}
    ):
        print(chunk)

# example with a single tool call
if __name__ == "__main__":
    import os
    asyncio.run(main())
