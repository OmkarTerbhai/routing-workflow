from langchain_openrouter import ChatOpenRouter;
from typing import TypedDict, List, Annotated;
import operator;
import asyncio;
from pydantic import BaseModel, Field;
from langgraph.types import Send;
from langgraph.graph import StateGraph, START, END;
from langchain_core.prompts import ChatPromptTemplate;
from dotenv import load_dotenv;

load_dotenv();

fanout_worker_llm = ChatOpenRouter(model="nvidia/nemotron-3-ultra-550b-a55b:free");
editor_base_llm = ChatOpenRouter(model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free");

class OverallState(TypedDict) :
    topic: str;
    subtopics: List[str];
    research_results: Annotated[List[str], operator.add];
    final_report: str

class ResearchState(TypedDict) :
    subtopic: str;

class SubtopicPlan(BaseModel) :
    subtopics: List[str] = Field(..., description="List of subtopics to research");

async def generate_subtopics(state:OverallState) :
    topic = state["topic"];

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Break the given research topic into clear, atomic, specific subtopics for a comprehensive report."),
            ("human", f"Topic: {topic}")
        ]
    );

    planner_chain = prompt | fanout_worker_llm.with_structured_output(SubtopicPlan);

    plan: SubtopicPlan = await planner_chain.ainvoke({
        "topic": topic
    });

    return {
      "subtopics": plan.subtopics
    };

async def research_individual_subtopics(state: ResearchState) :
    subtopic = state["subtopic"];

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Provide a concise, insightful 1 paragraph research summary of the following"),
        ("human", f"subtopic: {subtopic}")
    ]);

    researcher_chain = prompt | editor_base_llm;

    res = await researcher_chain.ainvoke({
        "subtopic": subtopic
    });

    findings = f"### {subtopic} ###\n {res.content}";

    return {
        "research_results": [findings]
    };

async def compile_final_report(state: OverallState) :
    combined_notes = "\n\n".join(state["research_results"]);

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior research editor. Synthesize the following subtopic research notes into a well-structured, cohesive executive report with an introduction and conclusion."),
        ("human", f"Main Topic: {state["topic"]}\n Subtopic Research Notes: {combined_notes}")
    ]);

    editor_chain = prompt | editor_base_llm;

    res = await editor_chain.ainvoke({
        "combined_notes": combined_notes
    });

    return {
        "final_report": res.content
    }

def map_to_subtopics(state: OverallState) :
    return [
        Send(
            "research_individual_subtopics",
            {"subtopic": subtopic}
        )
        for subtopic in state["subtopics"]
    ];


builder = StateGraph(OverallState);

builder.add_node("generate_subtopics", generate_subtopics);
builder.add_node("research_individual_subtopics", research_individual_subtopics);
builder.add_node("compile_final_report", compile_final_report);

builder.add_edge(START, "generate_subtopics");
builder.add_conditional_edges(
    "generate_subtopics",
    map_to_subtopics
);
builder.add_edge("research_individual_subtopics", "compile_final_report");
builder.add_edge("compile_final_report", END);

graph = builder.compile();

initial_state = {
    "topic": "Which animal assigns names to each other of the same species?",
    "subtopics": "",
    "research_results": [],
    "final_report": ""
};
async def run_graph() :
    res = await graph.ainvoke(initial_state);
    print(res);

asyncio.run(run_graph());