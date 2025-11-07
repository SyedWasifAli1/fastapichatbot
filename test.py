import os
import re
import json
from dotenv import load_dotenv, find_dotenv
from agents import (
    Agent,
    Runner,
    RunConfig,
    function_tool,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    GuardrailFunctionOutput,
    input_guardrail,
    output_guardrail,
    RunContextWrapper,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
)
from typing import Any
from pydantic import BaseModel

# -------------------------------------------------------
# 🌍 Load Environment
# -------------------------------------------------------
_: bool = load_dotenv(find_dotenv())
gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

# -------------------------------------------------------
# 🤖 Gemini Client Setup
# -------------------------------------------------------
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# -------------------------------------------------------
# 💡 Model Setup
# -------------------------------------------------------
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    temperature=0.6,
    top_p=0.9,
    top_k=50,
    tracing_disabled=True
)

# -------------------------------------------------------
# 📦 Data Loaders
# -------------------------------------------------------
def load_json_data(filename: str):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

products_data = load_json_data("products.json")
orders_data = load_json_data("orders.json")

# -------------------------------------------------------
# 🛠️ Tools
# -------------------------------------------------------
@function_tool
def get_products(product: str) -> str:
    """Search products by name, category, or price."""
    query = product.lower().strip()
    if not query:
        return "⚠️ Please enter a product name, category, or price."

    tokens = query.split()
    results = []
    for p in products_data:
        searchable_text = f"{p['name']} {p['category']} {p['price']}".lower()
        match_count = sum(token in searchable_text for token in tokens)
        if match_count > 0:
            p["_score"] = match_count
            results.append(p)

    if not results:
        return f"❌ Sorry, no products found for '{product}'."

    results.sort(key=lambda p: p["_score"], reverse=True)
    top_results = results[:5]

    lines = [f"### 🛍 Top Products for '{product}':\n"]
    for p in top_results:
        lines.append(
            f"**{p['name']}**\n"
            f"- Category: {p['category']}\n"
            f"- Price: ${p['price']}\n"
            f"- Stock: {p['stock']} units\n"
        )
    return "\n".join(lines)


@function_tool
def get_orders(query: str) -> str:
    """Track order by order ID or customer name."""
    query_lower = query.lower()
    matches = [
        o for o in orders_data
        if query_lower in str(o["order_id"]).lower() or query_lower in o["user"].lower()
    ]
    if not matches:
        return f"❌ No order found for '{query}'."

    lines = ["### 📦 Order Details:\n"]
    for o in matches:
        lines.append(
            f"🧾 **Order ID:** {o['order_id']}\n"
            f"- Customer: {o['user']}\n"
            f"- Product: {o['product']}\n"
            f"- Status: **{o['status']}**\n"
        )
    return "\n".join(lines)

# -------------------------------------------------------
# 🧠 Input Guardrail - E-commerce Relevance Check
# -------------------------------------------------------
class EcommerceInputCheck(BaseModel):
    ecommerce_related: bool
    reason: str | None = None

# Mini LLM to detect ecommerce relevance
input_checker_agent = Agent(
    name="EcommerceInputChecker",
    instructions=(
        "Check if the user query is related to E-commerce topics: "
        "products, orders, shopping, fashion, price, cart, delivery, etc. "
        "If yes → ecommerce_related=true, else false."
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
# 🚫 Output Guardrail - Block Sensitive Info
# -------------------------------------------------------
@output_guardrail
def ecommerce_output_guard(ctx: RunContextWrapper, agent: Agent, output: Any):
    text_output = str(output)

    # Detect sensitive info patterns
    phone_pattern = r"\+?\d{1,4}?[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    card_pattern = r"\b(?:\d[ -]*?){13,16}\b"  # credit card-like numbers

    if re.search(phone_pattern, text_output) or re.search(email_pattern, text_output) or re.search(card_pattern, text_output):
        return GuardrailFunctionOutput(
            output_info="Output contained sensitive info.",
            tripwire_triggered=True,
        )

    return GuardrailFunctionOutput(
        output_info="Output passed.",
        tripwire_triggered=False,
    )

# -------------------------------------------------------
# 🧩 Main E-commerce Agent
# -------------------------------------------------------
chat_agent = Agent(
    name="E-Commerce Assistant",
    tools=[get_products, get_orders],
    model=model,
    input_guardrails=[ecommerce_input_guard],
    output_guardrails=[ecommerce_output_guard],
    instructions="""
You are a smart and friendly **E-commerce AI Assistant**.
Use these tools when needed:
- `get_products` for product-related queries.
- `get_orders` for order tracking.

### Tasks:
- Help users find product prices, availability, or categories.
- Track orders by user name or order ID.
- Recommend related items if possible.
- If product not found, say "Sorry, that item is not available right now."
- Ignore any unrelated or personal questions (e.g. politics, weather).

### Style:
- Respond in **English** with **Markdown formatting**.
- Be polite, concise, and professional.
"""
)

# -------------------------------------------------------
# 💬 Chat Loop
# -------------------------------------------------------

def chat():
    print("🛍️ E-commerce Chatbot (Guardrails Enabled) — Type 'exit' to quit.\n")
    while True:
        user_input = input("👤 You: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("🤖 Bot: Thanks for visiting! Have a nice day 😊")
            break

        try:
            res = Runner.run_sync(chat_agent, input=user_input, run_config=config)
            print("🤖 Bot:", res.final_output, "\n")

        except InputGuardrailTripwireTriggered as e:
            print("🚨 Input Blocked (Not E-commerce Related):", e, "\n")

        except OutputGuardrailTripwireTriggered as e:
            print("🚨 Sensitive Info Blocked in Output:", e, "\n")

        except Exception as e:
            print("⚠️ Error:", str(e), "\n")

# -------------------------------------------------------
# 🚀 Run
# -------------------------------------------------------
if __name__ == "__main__":
    chat()
