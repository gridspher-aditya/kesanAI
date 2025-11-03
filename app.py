import os
import logging
import requests
import json
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime # Import datetime to get the current time

# --- LangChain Imports ---
from langchain_deepseek import ChatDeepSeek
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool, Tool
from langchain import hub

# --- 1. Load Environment Variables and Configure ---
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
API_BASE_URL = os.getenv("FARM_API_BASE_URL")
USERNAME = os.getenv("FARM_USERNAME")
PASSWORD = os.getenv("FARM_PASSWORD")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. Define the Custom Tool (Simplified) ---
@tool
def get_farm_data_by_device(device_id: int) -> str:
    """
    Fetches the most recent sensor data for a *specific device ID* from the farm API.
    Use this tool to answer any questions about the current weather, temperature,
    humidity, or rainfall for the given device ID.
    The input to this tool must be an integer (e.g., 1, 2, 3).
    """
    logger.info(f"Tool 'get_farm_data_by_device' triggered for device_id: {device_id}")
    
    # Use the new, simpler API endpoint
    data_url = f"https://gridsphere.in/dapi/?d_id={device_id}"

    try:
        headers = {
            'Accept': 'application/json, text/plain, */*',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        
        response = requests.get(data_url, headers=headers)
        response.raise_for_status() # Check for HTTP errors
        
        data = response.json()
        readings_list = data.get('readings', [])
        
        if isinstance(readings_list, list) and readings_list:
            # Return only the first item (the latest reading) from the list
            return json.dumps(readings_list[0])
        
        return json.dumps({"status": f"No readings available for device {device_id}"})

    except requests.exceptions.RequestException as e:
        logger.error(f"API interaction failed: {e}")
        return f"Error: Could not connect to the farm API. {e}"
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON from API response")
        return "Error: Received invalid data from the farm API."

# --- 3. Initialize Flask App and LangChain Agent ---
app = Flask(__name__)
CORS(app)

# --- NEW: Define the AI's persona and rules in a clear, constant string ---
SYSTEM_PERSONA = """
You are KeSAN, an AI farm assistant for Grid Sphere, specializing in Apple Orchards.
You are helpful, friendly, and an expert in farm data.

Follow these rules strictly on every response:
1.  **Formatting:** NEVER use markdown or any special symbols like *, #, or `.`
2.  **Language:** Respond ONLY in the language of the user's question (either Hindi or English).
3.  **Date Format:** When mentioning a date from a tool, parse the 'timestamp' (e.g., "2025-10-08 14:24:13") and format it as a full date (e.g., "8 October 2025").
4.  **Time Format:** Format all times in AM/PM (e.g., "2:24 PM").
5.  **Emojis:** Use relevant emojis (like 🌡️, 💧, ☀️, 🌬️, 🌧️) to make the response friendly.
6.  **Scope:** You can ONLY answer questions related to farm data and Apple farming. For any other questions, politely decline and state your purpose.

---

Give result in this format:
Hello! Here is your full weather report for the Apple Orchard, based on the latest data from your sensor.

📅 Date: [Date]
🕐 Time: [Time]

Current Conditions:

🌡️ Air Temperature: 
💧 Air Humidity: 
☀️ Light Intensity:
🌬️ Wind Speed: 
🧭 Wind Direction: [Eg: N, S, E, W, NE, SE, NW]
🌧️ Rainfall: 
📊 Pressure:

Soil Conditions:

🌱 Surface Temperature: 
💧 Surface Humidity: 
🌱 Depth Temperature: 
💧 Depth Humidity: 

[Final summary]

---
"""

agent_executor = None
try:
    llm = ChatDeepSeek(api_key=DEEPSEEK_API_KEY, model="deepseek-chat")
    
    tools = [get_farm_data_by_device]
    
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )
    
    logger.info("LangChain RAG Agent initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize LangChain agent: {e}")

# --- 4. Create the API Endpoint for the Chatbot ---
@app.route('/ask', methods=['POST'])
def ask_agent():
    if not agent_executor:
        return jsonify({"error": "AI agent is not available."}), 503

    data = request.get_json()
    user_question = data.get('question')
    device_id = data.get('deviceId')

    if not user_question or not device_id:
        return jsonify({"error": "A question and deviceId must be provided."}), 400

    # --- NEW: Get the current date and time to provide context to the AI ---
    # Example: "Monday, 3 November 2025 at 1:30 PM"
    current_time = datetime.now().strftime('%A, %d %B %Y at %I:%M %p')

    # --- NEW: Construct a more robust prompt ---
    prompt_with_context = f"""
    {SYSTEM_PERSONA}

    Current time for your reference: {current_time}
    The user is asking about device ID: {device_id}
    
    User's question: "{user_question}"
    """

    try:
        response = agent_executor.invoke({"input": prompt_with_context})
        final_answer = response.get('output', "Sorry, I couldn't find an answer.")
        return jsonify({"answer": final_answer})
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return jsonify({"error": "An internal error occurred while processing your question."}), 500

@app.route('/', methods=['GET'])
def start():
    return "Hello"

# --- 5. Run the Flask App ---
if __name__ == '__main__':
    app.run(debug=False)
