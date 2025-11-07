# from agents import (
#     Agent,
#     GuardrailFunctionOutput,
#     InputGuardrailTripwireTriggered,
#     OutputGuardrailTripwireTriggered,
#     RunContextWrapper,
#     Runner,
#     RunConfig,
#     function_tool,
#     OpenAIChatCompletionsModel,
#     AsyncOpenAI,
#     input_guardrail,
#     output_guardrail
# )
# import os
# import re
# import json
# from dotenv import load_dotenv
# from pydantic import BaseModel
# from typing import Any

# # -------------------------------------------------------
# # 🌍 Load environment and initialize API
# # -------------------------------------------------------
# load_dotenv()
# gemini_api_key = os.getenv("GEMINI_API_KEY")

# external_client = AsyncOpenAI(
#     api_key=gemini_api_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )

# # -------------------------------------------------------
# # 📦 JSON Data Loader
# # -------------------------------------------------------
# def load_json_data(filename: str):
#     try:
#         with open(filename, "r", encoding="utf-8") as f:
#             return json.load(f)
#     except FileNotFoundError:
#         print(f"⚠️ File not found: {filename}")
#         return []
#     except json.JSONDecodeError:
#         print(f"⚠️ Error decoding JSON in file: {filename}")
#         return []

# products_data = load_json_data("products.json")
# orders_data = load_json_data("orders.json")

# # -------------------------------------------------------
# # 🧰 Function Tools
# # -------------------------------------------------------
# @function_tool
# def get_products(product: str) -> str:
#     """Smart flexible search by name, category, or price from products.json."""
#     query = product.lower().strip()
#     if not query:
#         return "⚠️ Please enter a product name, category, or price to search."

#     tokens = query.split()
#     results = []

#     for p in products_data:
#         name = p["name"].lower()
#         category = p["category"].lower()
#         price_str = str(p["price"]).lower()

#         searchable_text = f"{name} {category} {price_str}"
#         match_count = sum(token in searchable_text for token in tokens)

#         if match_count > 0:
#             p["_score"] = match_count
#             results.append(p)

#     if not results:
#         return f"❌ Sorry, no products found for '{product}'."

#     results.sort(key=lambda p: p["_score"], reverse=True)
#     top_results = results[:5]

#     response_lines = [f"### 🛍 Top Matching Products for '{product}':\n"]
#     for p in top_results:
#         response_lines.append(
#             f"**{p['name']}**\n"
#             f"- Category: {p['category']}\n"
#             f"- Price: ${p['price']}\n"
#             f"- Stock: {p['stock']} units\n"
#         )
#     return "\n".join(response_lines)


# @function_tool
# def get_orders(query: str) -> str:
#     """Track order by order ID or customer name from orders.json."""
#     query_lower = query.lower()
#     matches = [
#         o for o in orders_data
#         if query_lower in str(o["order_id"]).lower() or query_lower in o["user"].lower()
#     ]
#     if not matches:
#         return f"No order found for '{query}'."
    
#     response_lines = []
#     for o in matches:
#         response_lines.append(
#             f"🧾 **Order ID:** {o['order_id']}\n"
#             f"- Customer: {o['user']}\n"
#             f"- Product: {o['product']}\n"
#             f"- Status: **{o['status']}**\n"
#         )
#     return "\n".join(response_lines)

# # -------------------------------------------------------
# # 🧠 Model Configuration
# # -------------------------------------------------------
# model = OpenAIChatCompletionsModel(
#     model="gemini-2.0-flash",
#     openai_client=external_client
# )

# config = RunConfig(
#     model=model,
#     model_provider=external_client,
#     tracing_disabled=True
# )

# # -------------------------------------------------------
# # 🛡️ Input Guardrail - Detect Non-Ecommerce Queries
# # -------------------------------------------------------
# class EcommerceInputCheck(BaseModel):
#     ecommerce_related: bool
#     reason: str | None = None

# input_checker_agent = Agent(
#     name="EcommerceInputChecker",
#        instructions=(
#         "Check if the user query is related to E-commerce topics: "
#         "products, orders, shopping, fashion, price, cart, delivery, etc. "
#         "If yes → ecommerce_related=true. "
#         "If the user query is a generic greeting, chat, or unrelated message like 'hello', "
#         "respond with ecommerce_related=false and reason='This AI assistant only handles product info and order tracking.'"
#     ),
#     model=model,
#     output_type=EcommerceInputCheck,
# )

# @input_guardrail
# async def ecommerce_input_guard(ctx: RunContextWrapper, agent: Agent, input):
#     res = await Runner.run(input_checker_agent, input)
#     print("\n[INPUT GUARDRAIL CHECK]", res.final_output)

#     return GuardrailFunctionOutput(
#         output_info=res.final_output.reason or "Input analyzed.",
#         tripwire_triggered=res.final_output.ecommerce_related is False,
#     )

# # -------------------------------------------------------
# # 🛡️ Output Guardrail - Block Sensitive Info
# # -------------------------------------------------------
# @output_guardrail
# def ecommerce_output_guard(ctx: RunContextWrapper, agent: Agent, output: Any):
#     text_output = str(output)

#     phone_pattern = r"\+?\d{1,4}?[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"
#     email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
#     card_pattern = r"\b(?:\d[ -]*?){13,16}\b"

#     if (
#         re.search(phone_pattern, text_output)
#         or re.search(email_pattern, text_output)
#         or re.search(card_pattern, text_output)
#     ):
#         return GuardrailFunctionOutput(
#             output_info="Output contained sensitive info.",
#             tripwire_triggered=True,
#         )

#     return GuardrailFunctionOutput(
#         output_info="Output passed.",
#         tripwire_triggered=False,
#     )

# # -------------------------------------------------------
# # 🧠 Chatbot Agent
# # -------------------------------------------------------
# chat_agent = Agent(
#     name="E-Commerce Assistant",
#     tools=[get_products, get_orders],
#     input_guardrails=[ecommerce_input_guard],
#     output_guardrails=[ecommerce_output_guard],
#     instructions="""
# You are a smart and friendly **E-commerce AI Assistant**.
# Use these tools when needed:
# - `get_products` for product-related queries.
# - `get_orders` for order tracking or order-related queries.

# ### Tasks:
# - Help users find product prices, availability, or categories.
# - Track orders by user name or order ID.
# - Recommend similar products when possible.
# - If product not found → say "Sorry, that item is not available right now."

# ### Style:
# - Answer in **English** with **Markdown formatting**.
# - Be short, polite, and professional.
# - Use bullet points and bold text.
# """
# )

# # -------------------------------------------------------
# # 💬 Chat Loop
# # -------------------------------------------------------
# def chat():
#     print("🛍️ E-commerce Chatbot (Guardrails Enabled) — Type 'exit' to quit.\n")
#     while True:
#         user_input = input("👤 You: ").strip()
#         if user_input.lower() in ["exit", "quit", "bye"]:
#             print("🤖 Bot: Thanks for visiting! Have a nice day 😊")
#             break

#         try:
#             res = Runner.run_sync(chat_agent, input=user_input, run_config=config)
#             print("🤖 Bot:", res.final_output, "\n")

#         except InputGuardrailTripwireTriggered as e:
#             print("🤖 Bot: Hi! I’m an E-commerce AI Assistant 🤖. I can only help you with **products** and **order tracking**.\n")


#         except OutputGuardrailTripwireTriggered as e:
#             print("🚨 Sensitive Info Blocked in Output:", e, "\n")

#         except Exception as e:
#             print("⚠️ Error:", str(e), "\n")

# # -------------------------------------------------------
# # 🚀 Run
# # -------------------------------------------------------
# if __name__ == "__main__":
#     chat()

import os
import re
import json
from dotenv import load_dotenv
from typing import Any
from pydantic import BaseModel
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    RunConfig,
    function_tool,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    input_guardrail,
    output_guardrail
)

# -------------------------------------------------------
# 🌍 Load environment and initialize API
# -------------------------------------------------------
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# -------------------------------------------------------
# 📦 JSON Data Loader
# -------------------------------------------------------
def load_json_data(filename: str):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ File not found: {filename}")
        return []
    except json.JSONDecodeError:
        print(f"⚠️ Error decoding JSON in file: {filename}")
        return []

products_data = load_json_data("products.json")
orders_data = load_json_data("orders.json")

# -------------------------------------------------------
# 🏷️ Output Structure Models
# -------------------------------------------------------
class ProductInfo(BaseModel):
    id: int
    name: str
    category: str
    price: float
    stock: int

class OrderInfo(BaseModel):
    order_id: str
    user: str
    product: str
    status: str

# -------------------------------------------------------
# 🧰 Function Tools
# -------------------------------------------------------
@function_tool
def get_products(product: str) -> list[ProductInfo]:
    """Return structured product info for top matches."""
    query = product.lower().strip()
    if not query:
        return []

    tokens = query.split()
    results = []

    for p in products_data:
        searchable_text = f"{p['name'].lower()} {p['category'].lower()} {str(p['price']).lower()}"
        match_count = sum(token in searchable_text for token in tokens)
        if match_count > 0:
            p["_score"] = match_count
            results.append(p)

    if not results:
        return []

    results.sort(key=lambda p: p["_score"], reverse=True)
    top_results = results[:5]

    structured_results = [ProductInfo(**{
        "id": p["id"],
        "name": p["name"],
        "category": p["category"],
        "price": p["price"],
        "stock": p["stock"]
    }) for p in top_results]

    return structured_results


@function_tool
def get_orders(query: str) -> list[OrderInfo]:
    """Return structured order info for matching orders."""
    query_lower = query.lower()
    matches = [
        o for o in orders_data
        if query_lower in str(o["order_id"]).lower() or query_lower in o["user"].lower()
    ]

    structured_results = [OrderInfo(**{
        "order_id": o["order_id"],
        "user": o["user"],
        "product": o["product"],
        "status": o["status"]
    }) for o in matches]

    return structured_results

# -------------------------------------------------------
# 🧠 Model Configuration
# -------------------------------------------------------
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# -------------------------------------------------------
# 🛡️ Input Guardrail - Detect Non-Ecommerce Queries
# -------------------------------------------------------
class EcommerceInputCheck(BaseModel):
    ecommerce_related: bool
    reason: str | None = None

input_checker_agent = Agent(
    name="EcommerceInputChecker",
    instructions=(
        "Check if the user query is related to E-commerce topics: "
        "products, orders, shopping, fashion, price, cart, delivery, etc. "
        "If yes → ecommerce_related=true. "
        "If the user query is a generic greeting, chat, or unrelated message like 'hello', "
        "respond with ecommerce_related=false and reason='This AI assistant only handles product info and order tracking.'"
    ),
    model=model,
    output_type=EcommerceInputCheck,
)

@input_guardrail
async def ecommerce_input_guard(ctx: RunContextWrapper, agent: Agent, input):
    res = await Runner.run(input_checker_agent, input)
    print("\n[INPUT GUARDRAIL CHECK]", res.final_output)

    return GuardrailFunctionOutput(
        output_info=res.final_output.reason or "Input analyzed.",
        tripwire_triggered=res.final_output.ecommerce_related is False,
    )

# -------------------------------------------------------
# 🛡️ Output Guardrail - Block Sensitive Info
# -------------------------------------------------------
@output_guardrail
def ecommerce_output_guard(ctx: RunContextWrapper, agent: Agent, output: Any):
    text_output = str(output)

    phone_pattern = r"\+?\d{1,4}?[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    card_pattern = r"\b(?:\d[ -]*?){13,16}\b"

    if (
        re.search(phone_pattern, text_output)
        or re.search(email_pattern, text_output)
        or re.search(card_pattern, text_output)
    ):
        return GuardrailFunctionOutput(
            output_info="Output contained sensitive info.",
            tripwire_triggered=True,
        )

    return GuardrailFunctionOutput(
        output_info="Output passed.",
        tripwire_triggered=False,
    )

# -------------------------------------------------------
# 🧠 Chatbot Agent
# -------------------------------------------------------
chat_agent = Agent(
    name="E-Commerce Assistant",
    tools=[get_products, get_orders],
    input_guardrails=[ecommerce_input_guard],
    output_guardrails=[ecommerce_output_guard],
    instructions="""
You are a smart and friendly **E-commerce AI Assistant**.
Use these tools when needed:
- `get_products` for product-related queries (returns structured ProductInfo list).
- `get_orders` for order tracking (returns structured OrderInfo list).

### Tasks:
- Help users find product prices, availability, or categories.
- Track orders by user name or order ID.
- Recommend similar products when possible.
- If product not found → say "Sorry, that item is not available right now."

### Style:
- Answer in **English** with **Markdown formatting**.
- Be short, polite, and professional.
- Use bullet points and bold text.
"""
)

# -------------------------------------------------------
# 💬 Chat Loop
# -------------------------------------------------------
def chat():
    print("🛍️ E-commerce Chatbot (Structured Output + Guardrails) — Type 'exit' to quit.\n")
    while True:
        user_input = input("👤 You: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("🤖 Bot: Thanks for visiting! Have a nice day 😊")
            break

        try:
            res = Runner.run_sync(chat_agent, input=user_input, run_config=config)
            
            # If tools returned structured outputs, display them
            if isinstance(res.final_output, list):
                for item in res.final_output:
                    if isinstance(item, ProductInfo):
                        print(f"🛍 Product: {item.name}, Category: {item.category}, Price: ${item.price}, Stock: {item.stock}")
                    elif isinstance(item, OrderInfo):
                        print(f"🧾 Order ID: {item.order_id}, User: {item.user}, Product: {item.product}, Status: {item.status}")
            else:
                print("🤖 Bot:", res.final_output, "\n")

        except InputGuardrailTripwireTriggered:
            print("🤖 Bot: Hi! I’m an E-commerce AI Assistant 🤖. I can only help you with **products** and **order tracking**.\n")

        except OutputGuardrailTripwireTriggered as e:
            print("🚨 Sensitive Info Blocked in Output:", e, "\n")

        except Exception as e:
            print("⚠️ Error:", str(e), "\n")

# -------------------------------------------------------
# 🚀 Run
# -------------------------------------------------------
if __name__ == "__main__":
    chat()
