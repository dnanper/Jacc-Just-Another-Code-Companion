"""Agent state schema - Foundation for entire agent.

Defines the state that flows through the LangGraph workflow.
Uses TypedDict with Annotated reducers for LangGraph compatibility.
"""

from typing import TypedDict, Literal, Annotated, Any, List
from operator import add
import time


class Message(TypedDict):
    """Single message in conversation history."""
    role: Literal["system", "user", "assistant"]
    content: str
    timestamp: float


class ExecutionResult(TypedDict):
    """Result from environment command execution."""
    returncode: int
    output: str
    timed_out: bool
    duration: float


class AgentState(TypedDict):
    """Main state passed through the LangGraph workflow.
    
    Design principles:
    - Append-only messages via reducer
    - Mutable working_memory for node scratch space
    - Clear separation: execution vs control vs metrics
    
    Attributes:
        messages: Linear conversation history (append-only)
        current_action: Parsed command from LLM response
        action_type: Type of action ("bash", "python", "submit", etc.)
        last_output: Result from last command execution
        working_memory: Short-term scratch space for nodes
        exit_status: None while running, set when finished
        exit_message: Explanation when agent exits
        step_count: Number of steps executed
        total_cost: Cumulative LLM cost in USD
        start_time: Unix timestamp when agent started
        codebase_retriever: Optional CodebaseRetriever for code context
        experience_store: Optional RepoExperienceStore for learned experiences
        retrieved_context: Context retrieved from codebase index
    """
    # Conversation - append-only via LangGraph reducer - no Overwrite
    messages: Annotated[list[Message], add]
    
    # Current step execution
    current_action: str | None
    action_type: Literal["bash", "python", "submit", "invalid"] | None
    
    # Execution result
    last_output: ExecutionResult | None
    
    # Working memory - extensible scratch space for nodes
    # Can store: retrieved context, parsed data, intermediate results
    working_memory: dict[str, Any]
    
    # Control flow
    exit_status: Literal["success", "error", "step_limit", "cost_limit"] | None
    exit_message: str | None
    
    # Metrics
    step_count: int
    total_cost: float
    start_time: float
    
    # HINDSIGHT
    global_guidelines: List[str]
    local_summary: str
    
    # Codebase indexing integration
    codebase_retriever: Any | None  # CodebaseRetriever instance
    experience_store: Any | None  # RepoExperienceStore instance
    retrieved_context: list[dict] | None  # Retrieved code context


def create_initial_state(
    system_message: str = "",
    task_message: str = "",
) -> AgentState:
    """Create a fresh initial state.
    
    Args:
        system_message: System prompt content
        task_message: User task/instance content
        
    Returns:
        Initialized AgentState ready for workflow
    """
    now = time.time()
    messages: list[Message] = []
    
    if system_message:
        messages.append({
            "role": "system",
            "content": system_message,
            "timestamp": now,
        })
    
    if task_message:
        messages.append({
            "role": "user", 
            "content": task_message,
            "timestamp": now,
        })
    
    return AgentState(
        messages=messages,
        current_action=None,
        action_type=None,
        last_output=None,
        working_memory={},
        exit_status=None,
        exit_message=None,
        step_count=0,
        total_cost=0.0,
        start_time=now,
        global_guidelines=[],
        local_summary="",
        codebase_retriever=None,
        experience_store=None,
        retrieved_context=None,
    )


def add_message(state: AgentState, role: str, content: str) -> dict:
    """Helper to create a message update for LangGraph.
    
    Returns dict that can be merged with state updates.
    """
    return {
        "messages": [{
            "role": role,
            "content": content,
            "timestamp": time.time(),
        }]
    }