---
title: "Getting Started with LangGraph: Build Your First Graph-Based LLM Application"
summary: "LangGraph simplifies building complex, stateful LLM applications by representing them as graphs. This tutorial guides you through creating a basic LangGraph application, covering setup, graph definition, state management, and execution."
keywords: ["LangGraph", "LLM", "graph", "stateful applications", "LangChain", "agents", "chains", "tutorial", "Python"]
created_at: "2025-11-09T10:13:45.615538"
reading_time_min: 7
status: draft
---

# Getting Started with LangGraph: Build Your First Graph-Based LLM Application

LangGraph simplifies the creation of complex, stateful LLM applications by representing them as graphs. This tutorial guides you through building a basic LangGraph application, covering setup, graph definition, state management, and execution.

## Introduction to LangGraph

LangGraph is a framework for building sophisticated LLM applications. It allows you to represent your application's logic as a graph, where nodes represent individual steps or components, and edges define the flow of execution between them.

**Why use LangGraph?**

*   **State Management:** LangGraph provides built-in state management, allowing you to track and update the application's state as it progresses through the graph. This is crucial for building conversational agents or applications that require memory.
*   **Complex Workflows:** LangGraph handles complex workflows with branching logic, loops, and parallel execution. You can define intricate application flows using graph structures.
*   **Modularity:** LangGraph promotes modularity by allowing you to break down your application into reusable components represented as nodes. This makes your code more organized and easier to maintain.
*   **Improved Debugging:** The graph representation makes it easier to visualize and debug your application's logic. You can trace the execution path and inspect the state at each node.

**LangChain Chains vs. LangGraph**

LangChain chains are suitable for simple LLM applications with a linear flow. However, when your application requires state management, branching logic, or complex workflows, LangGraph offers a more robust and flexible solution.

Think of LangChain chains as a straight road, while LangGraph is a network of roads with intersections, detours, and the ability to remember where you've been.

**Graph Concepts**

*   **Nodes:** Represent individual steps or components in your application. A node can be a LangChain LLM, a chain, a Python function, or any other callable object.
*   **Edges:** Define the flow of execution between nodes. Edges specify which node should be executed after another.
*   **State:** Represents the application's data at a given point in time. The state is typically a Python dictionary or dataclass that is passed between nodes and updated during execution.

## Prerequisites

Before building your first LangGraph application, ensure you have the following:

*   **Python Installation:** Python 3.8 or higher is recommended. You can download the latest version of Python from the official website: [https://www.python.org/downloads/](https://www.python.org/downloads/)
*   **Basic Understanding of LangChain:** Familiarity with LangChain concepts such as LLMs, prompts, and chains is helpful. You can learn more about LangChain at [https://www.langchain.com/](https://www.langchain.com/)
*   **Familiarity with LLMs and APIs:** You should have a basic understanding of how LLMs work and how to interact with them through APIs (e.g., OpenAI).
*   **Virtual Environment (Optional):** Using a virtual environment is recommended to isolate your project's dependencies. You can create a virtual environment using `venv` or `conda`.

    ```bash
    # Using venv
    python3 -m venv .venv
    source .venv/bin/activate

    # Using conda
    conda create -n langgraph_env python=3.9
    conda activate langgraph_env
    ```

## Installation and Setup

To get started with LangGraph, you need to install the necessary packages and set up your API keys.

### Installing LangGraph

Install `langgraph`, `langchain`, and `langchain-openai` using pip:

```bash
pip install langgraph langchain langchain-openai
```

### Setting up API Keys

You'll need an API key to access LLMs. For example, to use OpenAI's GPT model, you need to set the `OPENAI_API_KEY` environment variable.

```python
import os
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" # Replace with your actual API key
```

**Note:** It's best practice to store your API keys securely, such as using environment variables or a secrets management system.

### Importing Necessary Modules

Import the required modules from LangGraph and LangChain:

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
```

## Defining Your First Graph

Let's create a simple graph with two nodes: one for generating a question and another for answering it.

### Defining the Graph's State

The graph's state is a Python dictionary or dataclass that holds the data passed between nodes. For this example, we'll use a dictionary with a single key, `"messages"`, to store the conversation history.

```python
from typing import TypedDict, List, Dict, Any

class GraphState(TypedDict):
    messages: List[Dict[str, Any]]
```

### Creating Nodes

We'll create two nodes:

1.  **Question Generation Node:** This node uses an LLM to generate a question based on the current state.
2.  **Answering Node:** This node uses an LLM to answer the question based on the current state.

### Connecting Nodes with Edges

We'll connect the nodes with a directed edge, specifying that the answering node should be executed after the question generation node.

### Example: Simple Question-Answering Graph

```python
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Define the state
class GraphState(TypedDict):
    messages: List[Dict[str, Any]]

# 2. Define the nodes
def generate_question(state: GraphState):
    messages = state['messages']
    prompt = ChatPromptTemplate.from_template("Generate a question based on the following context: {messages}")
    model = ChatOpenAI()
    question_chain = prompt | model
    question = question_chain.invoke({"messages": messages})
    return {"messages": [{"role": "assistant", "content": question.content}]}


def answer_question(state: GraphState):
    messages = state['messages']
    prompt = ChatPromptTemplate.from_template("Answer the question based on the following context: {messages}")
    model = ChatOpenAI()
    answer_chain = prompt | model
    answer = answer_chain.invoke({"messages": messages})
    return {"messages": messages + [{"role": "assistant", "content": answer.content}]}

# 3. Create a new graph
graph = StateGraph(GraphState)

# 4. Add the nodes
graph.add_node("generate_question", generate_question)
graph.add_node("answer_question", answer_question)

# 5. Add the edges
graph.add_edge("generate_question", "answer_question")
graph.add_edge("answer_question", END)

# 6. Compile the graph
app = graph.compile()
```

## Implementing Nodes

Now, let's implement the question generation and answering nodes using LangChain LLMs.

### Creating Nodes using LangChain LLMs or Chains

You can use LangChain's LLMs and chains to define the logic within your nodes. This allows you to leverage LangChain's features for prompt engineering, model selection, and output parsing.

### Creating Nodes using Python Functions

You can also use Python functions to define the logic within your nodes. This is useful for implementing custom logic or integrating with external APIs.

### Accessing and Modifying the Graph's State

Within your nodes, you can access and modify the graph's state using the `state` argument. This allows you to update the state based on the node's output and pass data to subsequent nodes.

### Example: Implementing Question Generation and Answering Nodes

The example in the "Defining Your First Graph" section demonstrates implementing the nodes using Python functions that leverage LangChain's `ChatOpenAI` model and prompt templates.

## Defining Edges and Flow Control

Edges define the flow of execution between nodes. LangGraph provides different types of edges to control the flow of your application.

### Connecting Nodes with Directed Edges

Directed edges specify the order in which nodes should be executed. You can use the `add_edge` method to create directed edges.

### Using Conditional Edges for Branching Logic

Conditional edges allow you to create branching logic based on the graph's state. You can use the `add_conditional_edges` method to define conditional edges.

### Using `END` Node to Terminate the Graph Execution

The `END` node is a special node that terminates the graph execution. You can use the `add_edge` method to connect nodes to the `END` node.

### Example: Adding a Conditional Edge

This example demonstrates how to add a conditional edge to handle cases where the answer is not found. We'll assume the `answer_question` node returns a special value if it can't find an answer.

```python
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Define the state (if not already defined)
class GraphState(TypedDict):
    messages: List[Dict[str, Any]]

def route_by_answer(state: GraphState):
  messages = state['messages']
  last_message = messages[-1]["content"]
  if "I don't know" in last_message:
    return "generate_question" # Go back and ask another question
  else:
    return "end" # End the graph

# Assuming graph is already created and nodes are added
graph.add_conditional_edges("answer_question", route_by_answer, {"generate_question": "generate_question", "end": END})
```

## Compiling and Running the Graph

Once you've defined your graph, you need to compile it into an executable object and run it with an initial state.

### Compiling the Graph

Use the `compile` method to compile the graph into an executable object.

```python
app = graph.compile()
```

### Running the Graph with an Initial State

Use the `invoke` method to run the graph with an initial state.

```python
initial_state = {"messages": [{"role": "user", "content": "Tell me about the capital of France."}]}
result = app.invoke(initial_state)
print(result)
```

### Inspecting the Final State

After the graph execution, you can inspect the final state to see the results of the computation.

### Example: Compiling and Running the Question-Answering Graph

The previous code snippets demonstrate compiling and running the simple question-answering graph.

## Example: A More Complex Graph with State Management

Let's expand the initial example to include a memory component. We'll update the graph's state with each iteration and use the state to track conversation history.

```python
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Define the state
class GraphState(TypedDict):
    messages: List[Dict[str, Any]]
    conversation_history: str

# 2. Define the nodes
def generate_question(state: GraphState):
    messages = state['messages']
    prompt = ChatPromptTemplate.from_template("Generate a question based on the following context: {messages}.  Previous conversation history: {conversation_history}")
    model = ChatOpenAI()
    question_chain = prompt | model
    question = question_chain.invoke({"messages": messages, "conversation_history": state['conversation_history']})
    return {"messages": messages + [{"role": "assistant", "content": question.content}],
            "conversation_history": state['conversation_history'] + "\nAI: " + question.content}


def answer_question(state: GraphState):
    messages = state['messages']
    prompt = ChatPromptTemplate.from_template("Answer the question based on the following context: {messages}.  Previous conversation history: {conversation_history}")
    model = ChatOpenAI()
    answer_chain = prompt | model
    answer = answer_chain.invoke({"messages": messages, "conversation_history": state['conversation_history']})
    return {"messages": messages + [{"role": "assistant", "content": answer.content}],
            "conversation_history": state['conversation_history'] + "\nHuman: " + answer.content}

def route_by_answer(state: GraphState):
  messages = state['messages']
  last_message = messages[-1]["content"]
  if "I don't know" in last_message:
    return "generate_question" # Go back and ask another question
  else:
    return "end" # End the graph

# 3. Create a new graph
graph = StateGraph(GraphState)

# 4. Add the nodes
graph.add_node("generate_question", generate_question)
graph.add_node("answer_question", answer_question)

# 5. Add the edges
graph.add_conditional_edges("answer_question", route_by_answer, {"generate_question": "generate_question", "end": END})
graph.add_edge("generate_question", "answer_question")


# 6. Compile the graph
app = graph.compile()

# Run the graph
initial_state = {"messages": [{"role": "user", "content": "Tell me about the capital of France."}], "conversation_history": ""}
result = app.invoke(initial_state)
print(result)
```

## Debugging and Logging

LangGraph provides debugging tools and logging capabilities to help you identify and resolve errors in your graph.

### Using LangGraph's Built-in Debugging Tools

LangGraph's debugging tools allow you to step through the graph execution, inspect the state at each node, and identify any issues. Refer to LangChain documentation for the latest debugging features.

### Logging the State and Node Outputs

You can log the state and node outputs during execution to gain insights into the graph's behavior.  Use Python's built-in `logging` module or LangChain's tracing capabilities to log relevant information.

### Tips for Identifying and Resolving Errors

*   **Start with Simple Graphs:** Begin with a simple graph and gradually add complexity as you go.
*   **Test Your Nodes Individually:** Test each node in isolation to ensure it's working correctly.
*   **Use Logging:** Log the state and node outputs to track the flow of data and identify any unexpected behavior.
*   **Use Debugging Tools:** Utilize LangGraph's debugging tools to step through the graph execution and inspect the state.

## Conclusion

LangGraph is a framework for building complex, stateful LLM applications. By representing your application's logic as a graph, you can leverage LangGraph's features for state management, complex workflows, modularity, and improved debugging.

### Next Steps

*   **Explore More Advanced Features:** Explore LangGraph's more advanced features, such as agents and parallel execution.
*   **Experiment with Different Graph Structures:** Experiment with different graph structures to find the best way to represent your application's logic.
*   **Contribute to LangGraph:** Contribute to the LangGraph project by submitting bug reports, feature requests, or code contributions.

### Further Reading

*   **LangChain Documentation:** [https://python.langchain.com/](https://python.langchain.com/)
*   **LangGraph Documentation:** (Check the LangChain documentation for the latest LangGraph-specific pages)
*   **OpenAI API Documentation:** [https://platform.openai.com/docs/api-reference](https://platform.openai.com/docs/api-reference)
