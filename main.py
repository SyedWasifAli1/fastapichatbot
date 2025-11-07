# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
# import os
# from dotenv import load_dotenv
# import asyncio

# # Load environment
# load_dotenv()
# gemini_api_key = os.getenv("GEMINI_API_KEY")

# # Gemini client setup
# external_client = AsyncOpenAI(
#     api_key=gemini_api_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )

# # Model setup
# model = OpenAIChatCompletionsModel(
#     model="gemini-2.0-flash",
#     openai_client=external_client
# )

# # Run config
# config = RunConfig(
#     model=model,
#     model_provider=external_client,
#     tracing_disabled=True
# )

# # Agent setup (chatbot)
# chat_agent = Agent(
#     name="AI Assistant",
#     instructions="""
#     You are an intelligent and friendly AI Assistant.
#     Always respond **in English** using **Markdown format** for better readability.
    
#     ### Response Rules:
#     - Use **headings**, **bold text**, `code blocks`, and bullet points where helpful.
#     - Keep your answers clear, structured, and easy to read.
#     - For coding questions, provide clean, working examples inside fenced code blocks (```python).
#     - For explanations, break content into small sections with headings.
#     - Avoid unnecessary text or system messages.
    
#     ### Tone:
#     Be polite, confident, and helpful — like a senior developer or tech mentor guiding a student.
#     """
# )


# # FastAPI app
# app = FastAPI()

# # ✅ CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Request Model
# class ChatRequest(BaseModel):
#     prompt: str

# @app.get("/")
# def read_root():
#     return {"message": "🤖 AI Chatbot API running successfully!"}

# @app.post("/chat")
# async def chat_with_bot(request: ChatRequest):
#     loop = asyncio.get_event_loop()

#     def run_sync_in_new_loop():
#         new_loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(new_loop)
#         return Runner.run_sync(
#             chat_agent,
#             input=request.prompt,
#             run_config=config
#         )

#     try:
#         result = await loop.run_in_executor(None, run_sync_in_new_loop)
#         return {
#             "user_prompt": request.prompt,
#             "bot_response": result.final_output
#         }

#     except Exception as e:
#         return {"error": str(e)}
import os
import re
import json
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional, Any

from agents import (
    Agent,
    Runner,
    RunConfig,
    function_tool,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    input_guardrail,
    output_guardrail,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
)

# -------------------------------------------------------
# 🌍 Environment Setup
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
def load_json(filename: str):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

packages_data = load_json("packages.json")
gallery_data = load_json("gallery.json")
bookings_data = load_json("bookings.json")

# -------------------------------------------------------
# 🧱 Models
# -------------------------------------------------------
class PackageInfo(BaseModel):
    id: int
    name: str
    price: float
    duration: str
    features: list[str]

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

class AIResponse(BaseModel):
    packages: Optional[list[PackageInfo]] = None
    gallery: Optional[list[GalleryItem]] = None
    bookings: Optional[list[BookingInfo]] = None
    message: Optional[str] = None

# -------------------------------------------------------
# 🧰 Tools (Strict Token-based Search)
# -------------------------------------------------------
def token_match_score(query: str, text: str) -> int:
    tokens = query.lower().split()
    return sum(token in text.lower() for token in tokens)

@function_tool
def get_packages(query: str) -> list[PackageInfo]:
    """Search photography packages."""
    q = query.lower().strip()
    results = []
    for pkg in packages_data:
        searchable = f"{pkg['name']} {' '.join(pkg['features'])} {pkg['price']} {pkg['duration']}"
        score = token_match_score(q, searchable)
        if score > 0:
            pkg["_score"] = score
            results.append(pkg)
    results.sort(key=lambda x: x["_score"], reverse=True)
    return [PackageInfo(**{k: v for k, v in p.items() if k in PackageInfo.model_fields}) for p in results[:5]]

@function_tool
def get_gallery(query: str) -> list[GalleryItem]:
    """Return gallery items."""
    q = query.lower().strip()
    results = []
    for g in gallery_data:
        searchable = f"{g['title']} {g['category']}"
        score = token_match_score(q, searchable)
        if score > 0:
            g["_score"] = score
            results.append(g)
    results.sort(key=lambda x: x["_score"], reverse=True)
    return [GalleryItem(**{k: v for k, v in g.items() if k in GalleryItem.model_fields}) for g in results[:10]]

@function_tool
def get_bookings(query: str) -> list[BookingInfo]:
    """Check client booking info."""
    q = query.lower().strip()
    results = []
    for b in bookings_data:
        searchable = f"{b['client_name']} {b['package']} {b['status']}"
        score = token_match_score(q, searchable)
        if score > 0:
            b["_score"] = score
            results.append(b)
    results.sort(key=lambda x: x["_score"], reverse=True)
    return [BookingInfo(**{k: v for k, v in b.items() if k in BookingInfo.model_fields}) for b in results]

# -------------------------------------------------------
# 🧠 Model Config
# -------------------------------------------------------
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

# -------------------------------------------------------
# 🛡️ Guardrails
# -------------------------------------------------------
class PhotographyInputCheck(BaseModel):
    photography_related: bool
    reason: Optional[str] = None

input_checker_agent = Agent(
    name="PhotographyInputChecker",
    instructions=(
        "Determine if the user query is related to photography services such as "
        "packages, gallery, or bookings. "
        "If not related, set photography_related=false and provide a short reason."
    ),
    model=model,
    output_type=PhotographyInputCheck,
)

@input_guardrail
async def photography_input_guard(ctx: RunContextWrapper, agent: Agent, input):
    res = await Runner.run(input_checker_agent, input)
    return GuardrailFunctionOutput(
        output_info=res.final_output.reason or "Input checked.",
        tripwire_triggered=res.final_output.photography_related is False,
    )

@output_guardrail
def photography_output_guard(ctx: RunContextWrapper, agent: Agent, output: Any):
    text = str(output)
    patterns = [r"\b\d{13,16}\b", r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", r"\+?\d{1,4}?[-.\s]?\d{6,}"]
    if any(re.search(p, text) for p in patterns):
        return GuardrailFunctionOutput("Sensitive info detected.", True)
    return GuardrailFunctionOutput("Clean output.", False)

# -------------------------------------------------------
# 🤖 Photography Assistant Agent (Strict Tool-Only)
# -------------------------------------------------------
photo_agent = Agent(
    name="Photography Assistant",
    tools=[get_packages, get_gallery, get_bookings],
    input_guardrails=[photography_input_guard],
    output_guardrails=[photography_output_guard],
    output_type=AIResponse,
    instructions="""
You are **Photography Studio AI Assistant** 📸.

🧠 Role:
- You ONLY answer using the provided tools (get_packages, get_gallery, get_bookings).
- You never create or guess data yourself.

📦 If user asks about:
- "package", "price", "features" → use get_packages
- "gallery", "photo", "category" → use get_gallery
- "booking", "client", "status" → use get_bookings

💬 If user asks general or greeting:
Respond ONLY with message:
"I’m a Photography Studio Assistant! I can help you find packages, view gallery photos, or check booking details."

⚠️ Important:
- Never make up answers.
- Always return a valid AIResponse object.
- Do not return plain text.
"""
)

# -------------------------------------------------------
# 🚀 FastAPI Setup
# -------------------------------------------------------
app = FastAPI(title="Photography Studio AI Assistant")

class QueryInput(BaseModel):
    query: str

@app.post("/chat")
async def chat_with_ai(data: QueryInput):
    query = data.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query not allowed")

    start_time = time.perf_counter()
    try:
        res = await Runner.run(photo_agent, input=query, run_config=config)
        elapsed = int((time.perf_counter() - start_time) * 1000)
        output = res.final_output

        # Always enforce AIResponse structure
        if not isinstance(output, AIResponse):
            output = AIResponse(message="I’m a Photography Studio Assistant! I can help with packages, gallery, or booking info.")

        return {
            "latency_ms": elapsed,
            "bot": "📸 Photography Studio AI Assistant",
            "response": output.model_dump(exclude_none=True)
        }

    except InputGuardrailTripwireTriggered:
        return {
            "latency_ms": int((time.perf_counter() - start_time) * 1000),
            "bot": "📸 Photography Studio AI Assistant",
            "response": {"message": "I’m a Photography Studio Assistant! I can help with packages, gallery, or booking info."}
        }

    except OutputGuardrailTripwireTriggered:
        return {"bot": "🚫 Sensitive information detected and blocked."}

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Photography AI Assistant running"}
