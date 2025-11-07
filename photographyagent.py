import os
import re
import json
import asyncio
from dotenv import load_dotenv
from typing import Optional, Any, List
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
    output_guardrail,
    set_tracing_disabled
)

# -------------------------------------------------------
# 🌍 Load Environment
# -------------------------------------------------------
load_dotenv()
set_tracing_disabled(disabled=True)
gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# -------------------------------------------------------
# 📦 JSON Data Loader
# -------------------------------------------------------
def load_json(filename: str):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ File not found: {filename}")
        return []
    except json.JSONDecodeError:
        print(f"⚠️ Error decoding JSON: {filename}")
        return []

packages_data = load_json("packages.json")
gallery_data = load_json("gallery.json")
bookings_data = load_json("bookings.json")

# -------------------------------------------------------
# 🧱 Output Data Models
# -------------------------------------------------------
class PackageInfo(BaseModel):
    id: int
    name: str
    price: float
    duration: str
    features: List[str]

class GalleryItem(BaseModel):
    id: int
    title: str
    category: str
    image_url: str

class BookingInfo(BaseModel):
    id: int
    client_name: str
    package: str
    date: str
    status: str

# -------------------------------------------------------
# 🧰 Tools
# -------------------------------------------------------
@function_tool
def get_packages(query: str) -> List[PackageInfo]:
    """Search for photography packages by name or features."""
    q = query.lower().strip()
    if not q:
        return []
    tokens = q.split()
    results = []
    for pkg in packages_data:
        searchable = f"{pkg['name'].lower()} {' '.join(pkg['features']).lower()} {pkg['price']}"
        match_count = sum(token in searchable for token in tokens)
        if match_count > 0:
            pkg["_score"] = match_count
            results.append(pkg)
    if not results:
        return []
    results.sort(key=lambda x: x["_score"], reverse=True)
    top = results[:5]
    return [PackageInfo(**{
        "id": p["id"],
        "name": p["name"],
        "price": p["price"],
        "duration": p["duration"],
        "features": p["features"]
    }) for p in top]

@function_tool
def get_gallery(category: str) -> List[GalleryItem]:
    """Return gallery images by category (e.g., wedding, event, portrait)."""
    q = category.lower().strip()
    matches = [g for g in gallery_data if q in g["category"].lower() or q in g["title"].lower()]
    return [GalleryItem(**{
        "id": g["id"],
        "title": g["title"],
        "category": g["category"],
        "image_url": g["image_url"]
    }) for g in matches[:10]]

@function_tool
def get_bookings(client: str) -> List[BookingInfo]:
    """Check booking status by client name."""
    q = client.lower().strip()
    matches = [b for b in bookings_data if q in b["client_name"].lower()]
    return [BookingInfo(**{
        "id": b["id"],
        "client_name": b["client_name"],
        "package": b["package"],
        "date": b["date"],
        "status": b["status"]
    }) for b in matches]

# -------------------------------------------------------
# 🧠 Model Config
# -------------------------------------------------------
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

# -------------------------------------------------------
# 🛡️ Input Guardrail
# -------------------------------------------------------
class PhotographyInputCheck(BaseModel):
    photography_related: bool
    reason: Optional[str] = None

input_checker_agent = Agent(
    name="PhotographyInputChecker",
    instructions=(
        "Check if user query is related to photography topics like packages, "
        "photoshoots, gallery, booking, photography styles, or studio services. "
        "If yes → photography_related=true. "
        "If not → photography_related=false and reason='This assistant only helps with packages, gallery, and booking details.'"
    ),
    model=model,
    output_type=PhotographyInputCheck,
)

@input_guardrail
async def photography_input_guard(ctx: RunContextWrapper, agent: Agent, input: str):
    res = await Runner.run(input_checker_agent, input)
    print("\n[INPUT GUARDRAIL CHECK]", res.final_output)
    return GuardrailFunctionOutput(
        output_info=res.final_output.reason or "Input analyzed.",
        tripwire_triggered=res.final_output.photography_related is False,
    )

# -------------------------------------------------------
# 🛡️ Output Guardrail
# -------------------------------------------------------
@output_guardrail
def photography_output_guard(ctx: RunContextWrapper, agent: Agent, output: Any):
    text_output = str(output)
    patterns = [
        r"\+?\d{1,4}?[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}",
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        r"\b(?:\d[ -]*?){13,16}\b"
    ]
    if any(re.search(p, text_output) for p in patterns):
        return GuardrailFunctionOutput(
            output_info="Sensitive info blocked.",
            tripwire_triggered=True
        )
    return GuardrailFunctionOutput(
        output_info="Output clean.",
        tripwire_triggered=False
    )

# -------------------------------------------------------
# 🤖 Main AI Agent
# -------------------------------------------------------
photo_agent = Agent(
    name="Photography Assistant",
    tools=[get_packages, get_gallery, get_bookings],
    input_guardrails=[photography_input_guard],
    output_guardrails=[photography_output_guard],
    instructions="""
You are a friendly and intelligent **Photography Studio AI Assistant** 🤖.

Your purpose is to help users with:
- 📸 Photography Packages (pricing, duration, and features)
- 🖼️ Gallery photos by category (wedding, event, portrait, etc.)
- 📅 Booking details and availability (client name, date, status)

Always return a **JSON** in this format:
{
  "bot": "response message",
  "data": { 
    "packages": [... or null],
    "gallery": [... or null],
    "bookings": [... or null]
  }
}
"""
)

# -------------------------------------------------------
# 💬 Chat Interface
# -------------------------------------------------------
def chat():
    print("📸 Photography Studio AI Assistant — JSON Mode")
    while True:
        user_input = input("👤 You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print(json.dumps({
                "bot": "Thanks for visiting! Have a great photoshoot 📸",
                "data": None
            }, indent=2, ensure_ascii=False))
            break

        try:
            res = Runner.run_sync(photo_agent, input=user_input, run_config=config)

            payload = {
                "packages": None,
                "gallery": None,
                "bookings": None
            }

            if isinstance(res.final_output, list) and res.final_output:
                if isinstance(res.final_output[0], PackageInfo):
                    payload["packages"] = [i.model_dump() for i in res.final_output]
                elif isinstance(res.final_output[0], GalleryItem):
                    payload["gallery"] = [i.model_dump() for i in res.final_output]
                elif isinstance(res.final_output[0], BookingInfo):
                    payload["bookings"] = [i.model_dump() for i in res.final_output]

            print(json.dumps({
                "bot": "Here’s the requested information 📸",
                "data": payload
            }, indent=2, ensure_ascii=False))

        except InputGuardrailTripwireTriggered:
            print(json.dumps({
                "bot": "Hi there! I can help you with packages, gallery, or booking details only.",
                "data": None
            }, indent=2, ensure_ascii=False))

        except OutputGuardrailTripwireTriggered:
            print(json.dumps({
                "bot": "🚫 Sensitive info blocked in output.",
                "data": None
            }, indent=2, ensure_ascii=False))

        except Exception as e:
            print(json.dumps({
                "bot": "⚠️ Internal error occurred.",
                "error": str(e),
                "data": None
            }, indent=2, ensure_ascii=False))


# -------------------------------------------------------
# 🚀 Run
# -------------------------------------------------------
if __name__ == "__main__":
    chat()
