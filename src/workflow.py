from typing import TypedDict;
from langgraph.graph import StateGraph, START, END;

class State(TypedDict) :
    topic: str;
    draft: str;
    feedback: str;
    revisions: int;


def write_draft(state: State) -> dict:
    print(f"[write draft] writing first draft");
    return {"draft": f"A short note about {state["topic"]}.",
            "revisions": 0};

def review(state: State) -> dict :
    print("[review draft] Reviewing draft");

    if len(state["draft"]) < 100 :
        return {"feedback": "Too short", "revisions": state["revisions"] + 1};

    return {"feedback": "Approved", "revisions": state["revisions"] + 1};

def publish(state: State) -> dict:
    print(f"[publish] FINAL: {state['draft']}")
    return {};

def route_after_review(state: State) -> str :
    if state["feedback"] == "Approved" :
        return "publish";

    return "revise";

def revise(state: State) -> dict:
    print("[revise] improving draft")
    return {
        "draft": state["draft"] + " It covers the key ideas in more detail.",
        "revisions": state["revisions"] + 1,
    }


builder = StateGraph(State);

builder.add_node("write_draft", write_draft);
builder.add_node("review_draft", review);
builder.add_node("publish", publish);
builder.add_node("revise", revise);

builder.add_edge(START, "write_draft");
builder.add_edge("write_draft", "review_draft");
builder.add_conditional_edges(
    "review_draft", route_after_review,
    {"revise": "revise", "publish": "publish"}
);
builder.add_edge("revise", "review_draft");

builder.add_edge("publish", END);

graph = builder.compile();

res = graph.invoke({"topic": "LangGraph", "draft": "", "feedback": "", "revisions": 0});
print(res);
