import os
import json
import re
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
# 📦 JSON Loader
# -------------------------------------------------------
def load_json_data(filename: str):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ File not found: {filename}")
        return []
    except json.JSONDecodeError:
        print(f"⚠️ Invalid JSON: {filename}")
        return []

packages_data = load_json_data("packages.json")

# -------------------------------------------------------
# 🏷️ Model
# -------------------------------------------------------
class PackageInfo(BaseModel):
    id: int
    name: str
    price: float
    duration: str
    features: list[str]

# -------------------------------------------------------
# 🧰 Function Tool (Voice/Token Search)
# -------------------------------------------------------
@function_tool
def get_packages(query: str) -> list[PackageInfo]:
    """
    Return matching photography packages.
    Voice-style token search — works with partial keywords like 'wedding', 'drone', 'cinematic'.
    """
    if not query:
        return []
    q_tokens = query.lower().split()

    results = []
    for p in packages_data:
        searchable_text = (
            f"{p['name'].lower()} {p['duration'].lower()} {' '.join(p['features']).lower()}"
        )
        score = sum(token in searchable_text for token in q_tokens)
        if score > 0:
            p["_score"] = score
            results.append(p)

    results.sort(key=lambda x: x["_score"], reverse=True)
    top = results[:5]

    return [
        PackageInfo(
            id=p["id"],
            name=p["name"],
            price=p["price"],
            duration=p["duration"],
            features=p["features"]
        ) for p in top
    ]

# -------------------------------------------------------
# 🧠 Model Config
# -------------------------------------------------------
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)
config = RunConfig(model=model, model_provider=external_client, tracing_disabled=True)

# -------------------------------------------------------
# 🛡️ Input Guardrail
# -------------------------------------------------------
class PhotographyInputCheck(BaseModel):
    photography_related: bool
    reason: str | None = None

input_checker_agent = Agent(
    name="PhotographyInputChecker",
    model=model,
    output_type=PhotographyInputCheck,
    instructions=(
        "Check if the user's query relates to photography packages, shoots, or pricing. "
        "If not → photography_related=false and reason='I can only assist with photography packages or services.'"
    )
)

@input_guardrail
async def photography_input_guard(ctx: RunContextWrapper, agent: Agent, input):
    res = await Runner.run(input_checker_agent, input)
    print("\n[INPUT GUARDRAIL CHECK]", res.final_output)
    return GuardrailFunctionOutput(
        output_info=res.final_output.reason or "Input checked.",
        tripwire_triggered=res.final_output.photography_related is False,
    )

# -------------------------------------------------------
# 🛡️ Output Guardrail
# -------------------------------------------------------
@output_guardrail
def photography_output_guard(ctx: RunContextWrapper, agent: Agent, output: Any):
    text_output = str(output)
    sensitive_patterns = [
        r"\+?\d{1,4}?[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}",
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    ]
    for pat in sensitive_patterns:
        if re.search(pat, text_output):
            return GuardrailFunctionOutput("Sensitive info detected.", True)
    return GuardrailFunctionOutput("Output clean.", False)

# -------------------------------------------------------
# 🤖 Photography Assistant Agent
# -------------------------------------------------------
photo_agent = Agent(
    name="Photography Studio Assistant",
    tools=[get_packages],
    input_guardrails=[photography_input_guard],
    output_guardrails=[photography_output_guard],
    instructions="""
You are a professional **Photography Studio AI Assistant** 🎞️.
Use `get_packages` to show package details (name, price, duration, features).

### Tasks:
- Understand both text & voice-style searches like "show wedding package" or "drone shoot package".
- If no results → say "Sorry, no matching package found."
- Show results as Markdown bullet list with bold names & clear formatting.

### Style:
- Single clean Markdown block.
- Friendly, modern, professional tone.
"""
)

# -------------------------------------------------------
# 💬 Chat Loop
# -------------------------------------------------------
def chat():
    print("📸 Photography Studio Chatbot — Type 'exit' to quit.\n")
    while True:
        user_input = input("👤 You: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("🤖 Bot: Thanks for visiting our studio! 📷 Have a great day!")
            break

        try:
            res = Runner.run_sync(photo_agent, input=user_input, run_config=config)
            if isinstance(res.final_output, list):
                formatted = "\n\n".join([
                    f"**{p.name}** — ${p.price} ({p.duration})\n"
                    + "\n".join([f"• {f}" for f in p.features])
                    for p in res.final_output
                ])
                print(f"\n🤖 Bot:\n{formatted}\n")
            else:
                print(f"\n🤖 Bot: {res.final_output}\n")

        except InputGuardrailTripwireTriggered:
            print("\n🤖 Bot: I can only help with **photography packages or shoots**.\n")

        except OutputGuardrailTripwireTriggered:
            print("\n🚨 Sensitive info blocked.\n")

        except Exception as e:
            print("\n⚠️ Error:", str(e), "\n")

# -------------------------------------------------------
# 🚀 Run
# -------------------------------------------------------
if __name__ == "__main__":
    chat()
