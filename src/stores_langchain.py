from langgraph.graph import MessagesState, StateGraph, START, END;
from typing import TypedDict;
from langchain_core.runnables import RunnableConfig;
from langgraph.runtime import Runtime;
from langchain_core.messages import AIMessage, HumanMessage;
from langgraph.checkpoint.memory import InMemorySaver;
from langgraph.store.memory import InMemoryStore;

class AgentState(MessagesState) :
    intent: str;
    order_id: str;
    preferred_language: str;

class Context(TypedDict) :
    user_id: str;
    plan: str;

def parse_request(state: AgentState, config: RunnableConfig) -> dict :
    user_txt = str(state["messages"][-1].content).strip();
    thread_id = config["configurable"]["thread_id"];

    print(f"[Node 1] thread_id = {thread_id}");
    print(f"[Node 1] message_count = {len(state["messages"])}");
    print(f"[Node 1] existing_order = {state.get('order_id', 'N/A')}");

    if user_txt.startswith("My Order ID is") :
        order_id = user_txt.removeprefix("My Order ID is").strip();

        return {
            "order_id": order_id,
            "intent": "save_order"
        };

    if user_txt == "Where is my order?" :
        return {"intent": "ask_order"};

    if user_txt.startswith("Remember my language is ") :
        return {"intent": "save_language"};

    if user_txt == "Which language do I prefer?" :
        return {"intent": "ask_language"};

    return {"intent": "unknown"};

def load_user_memory(state: AgentState,
                     runtime: Runtime[Context]) -> dict :
    user_id = runtime.context.get("user_id");
    plan = runtime.context.get("plan");

    store = runtime.store;

    namespace = ("users", user_id);

    if state["intent"] == "save_language" :
        message = str(state["messages"][-1].content).strip();
        language = message.removeprefix("Remember my language is ").strip();
        store.put(
            namespace,
            "language",
            {"value": language}
        );

    if state["intent"] == "save_order" :
        message = str(state["messages"][-1].content).strip();
        order = message.removeprefix("My Order ID is ").strip();
        store.put(
            namespace,
            "order",
            {"value": order}
        );

    memory_lang = store.get(namespace, "language");
    memory_ord = store.get(namespace, "order");

    language = (
        memory_lang.value["value"]
        if memory_lang is not None else "Not Set"
    );

    order = (
        memory_ord.value["value"]
        if memory_ord is not None else "Not Set"
    );

    if state["intent"] == "ask_order" :
        message = str(state["messages"][-1].content).strip();
        return {
          "order_id": order
        };

    if state["intent"] == "ask_language" :
        return {
            "preferred_language": language
        };

def respond(
        state: AgentState,
        config: RunnableConfig,
        runtime: Runtime[Context]
) -> dict :
    thread_id = config["configurable"]["thread_id"];
    user_id = runtime.context.get("user_id");
    intent = state["intent"];

    if intent == "save_order" :
        answer = (
            f"Saved order {state["order_id"]} for user {user_id} "
            f"in thread {thread_id}"
        );

    elif intent == "ask_order" :
        if state.get("order_id") :
            answer = (
                f"Your order {state["order_id"]} is being processed"
                f" It will be shipped soon"
            );

        else :
            answer = (
                f"Could not find your order"
            );

    elif intent == "save_language" :
        answer = (
            f"Saved your preferred language {state["preferred_language"]}"
        );

    elif intent == "ask_language" :
        if state.get("prefered_language") :
            answer = (
                f"Your preferred language is {state['preferred_language']}"
            );

        else :
            answer = (
                 "You haven't set your preferred language yet. "
                "Please set your preferred language."
            );

    else:
        answer = "Ask about your order or language preferences."

    return {
        "messages": [AIMessage(content=answer)]
    }


builder = StateGraph(AgentState, context_schema=Context);

builder.add_node("parse_request", parse_request);
builder.add_node("load_user_memory", load_user_memory);
builder.add_node("respond", respond);

builder.add_edge(START,"parse_request");
builder.add_edge("parse_request", "load_user_memory");
builder.add_edge("load_user_memory", "respond");
builder.add_edge("respond", END);

checkpointer = InMemorySaver();
store = InMemoryStore();

graph = builder.compile(
    checkpointer=checkpointer,
    store=store
);

def chat(user_id, thread_id, message, plan="free"):

    print(f"\nUser {user_id} | Thread {thread_id}")
    print(f"\nInput: {message}")

    result = graph.invoke({
        "messages": [HumanMessage(content=message)]
    },
    config={"configurable": {"thread_id": thread_id}, "tags": ["chat", "demo"]},
    context={"user_id": user_id, "plan": plan}
    )

    # graph has been invoked successfully
    print("Answer: ", result["messages"][-1].content)

    print("Total messages: ", len(result["messages"]))

    return result

chat("User_A", "a1", "My Order ID is 123", "premium");

chat("User_A", "a1", "Where is my order?", "premium");






