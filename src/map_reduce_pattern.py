from typing import TypedDict, List, Annotated;
import operator;
from langgraph.types import Send;
from langgraph.graph import StateGraph, START, END;

class SuperState(TypedDict) :
    cart_id: str;
    raw_items: List[dict];
    final_receipt: str;
    processed_items: Annotated[List[str], operator.add];
    total_discount: Annotated[float, operator.add];
    total_amount: Annotated[float, operator.add];

class CartItemWorkerState(TypedDict) :
    item_name: str;
    price: float;


def fetch_cart_items(state: SuperState) :
    items_from_db = [
        {"item_name": "Apple Macbook Pro", "price": 2000},
        {"item_name": "Apple Macbook Air", "price": 1500},
        {"item_name": "Apple Mac Mini", "price": 1000},
        {"item_name": "Apple Iphone 15", "price": 1000}
    ];

    return {
        "raw_items": items_from_db
    }

def fanout_calculate_item_total(state: CartItemWorkerState) :
    name = state["item_name"];
    price = state["price"];

    final_price = (price * 0.9) * 1.05;

    summary = f"{name} : ${final_price}";

    return {
        "processed_items": [summary],
        "total_discount": price * 0.1,
        "total_amount": final_price
    }


def generate_receipt(state: SuperState):
    lines = ["=" * 40, f"RECEIPT FOR CART #{state['cart_id']}", "=" * 40]

    for idx, line in enumerate(state["processed_items"], start=1):
        lines.append(f"{idx}. {line}")

    lines.append("=" * 40)

    lines.append(f"Total Discount: ${state['total_discount']}")
    lines.append(f"Total Amount: ${state['total_amount']}")
    return {
        "final_receipt": "\n".join(lines)
    }

def map_items_to_fanout_nodes(state: SuperState) :
    return [
        Send(
            "fanout_calculate_item_total",
            {"item_name": item["item_name"],
                   "price": item["price"]}
        )
        for item in state["raw_items"]
    ];

builder = StateGraph(SuperState);

builder.add_node("fetch_cart_items", fetch_cart_items);
builder.add_node("fanout_calculate_item_total", fanout_calculate_item_total);
builder.add_node("generate_receipt", generate_receipt);

builder.add_edge(START, "fetch_cart_items");
builder.add_conditional_edges(
    "fetch_cart_items",
    map_items_to_fanout_nodes
);
builder.add_edge("fanout_calculate_item_total", "generate_receipt");
builder.add_edge("generate_receipt", END);

graph = builder.compile();

initial_state = {"cart_id": "1234567890", "raw_items": [], "final_receipt": "", "processed_items": [], "total_discount": 0, "total_amount": 0};

res = graph.invoke(initial_state);
print(res["final_receipt"]);