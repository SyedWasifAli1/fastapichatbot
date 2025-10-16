from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
import os
from dotenv import load_dotenv
import asyncio

# Load environment
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Gemini client setup
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Model setup
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Run config
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Agent setup (chatbot)
chat_agent = Agent(
    name="AI Assistant",
    instructions="""
    You are an intelligent and friendly AI Assistant.
    Always respond **in English** using **Markdown format** for better readability.
    
    ### Response Rules:
    - Use **headings**, **bold text**, `code blocks`, and bullet points where helpful.
    - Keep your answers clear, structured, and easy to read.
    - For coding questions, provide clean, working examples inside fenced code blocks (```python).
    - For explanations, break content into small sections with headings.
    - Avoid unnecessary text or system messages.
    
    ### Tone:
    Be polite, confident, and helpful — like a senior developer or tech mentor guiding a student.
    """
)


# FastAPI app
app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Model
class ChatRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"message": "🤖 AI Chatbot API running successfully!"}

@app.post("/chat")
async def chat_with_bot(request: ChatRequest):
    loop = asyncio.get_event_loop()

    def run_sync_in_new_loop():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return Runner.run_sync(
            chat_agent,
            input=request.prompt,
            run_config=config
        )

    try:
        result = await loop.run_in_executor(None, run_sync_in_new_loop)
        return {
            "user_prompt": request.prompt,
            "bot_response": result.final_output
        }

    except Exception as e:
        return {"error": str(e)}
