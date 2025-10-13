import os
import logging
import requests
import json
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- LangChain Imports ---
from langchain_deepseek import ChatDeepSeek
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain import hub

# --- 1. Load Environment Variables and Configure ---
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
API_BASE_URL = os.getenv("FARM_API_BASE_URL")
USERNAME = os.getenv("FARM_USERNAME")
PASSWORD = os.getenv("FARM_PASSWORD")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. Define the Custom Tool with Full Authentication Logic ---
def get_latest_farm_data(input: str = "") -> str:
    """
    You are a farm agent called KeSAN. You can only give answer related to farm data, nothing else.
    Fetches the most recent sensor data from the farm API after authenticating.
    Use this tool for any questions about the current weather, temperature,
    humidity, or rainfall on the farm.
    """
    logger.info("Tool 'get_latest_farm_data' triggered.")
    if not all([API_BASE_URL, USERNAME, PASSWORD]):
        return "Error: API credentials are not configured."

    with requests.Session() as session:
        try:
            headers = {
                'Accept': 'application/json, text/plain, */*',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
            }
            # Step 1: Get CSRF Token
            csrf_response = session.get(f"{API_BASE_URL}/getCSRF", headers=headers)
            csrf_response.raise_for_status()
            csrf_data = csrf_response.json()
            csrf_name, csrf_value = csrf_data['csrf_name'], csrf_data['csrf_token']
            
            # Step 2: Log In
            login_data = {'username': USERNAME, 'password': PASSWORD, csrf_name: csrf_value}
            login_response = session.post(f"{API_BASE_URL}/login", data=login_data, headers=headers)
            login_response.raise_for_status()

            if "Login successful" not in login_response.text and "error" in login_response.text.lower():
                 raise Exception(f"Login failed: {login_response.text}")

            # Step 3: Fetch Live Farm Data
            data_url = f"{API_BASE_URL}/live-data/2" # Assuming device_id 1
            data_response = session.get(data_url, headers=headers)
            data_response.raise_for_status()
            
            # The API response has a 'data' key which contains a list
            live_data_list = data_response.json().get('data', [])
            
            if isinstance(live_data_list, list) and live_data_list:
                # Return only the first (and likely only) item in the list
                return json.dumps(live_data_list[0])
            
            return json.dumps({"status": "No live data available"})
        except requests.exceptions.RequestException as e:
            logger.error(f"API interaction failed: {e}")
            return f"Error: Could not connect to the farm API. {e}"

# --- 3. Initialize Flask App and LangChain Agent ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing for your React app

agent_executor = None
try:
    llm = ChatDeepSeek(api_key=DEEPSEEK_API_KEY, model="deepseek-chat")
    tools = [
        Tool(
            name="get_latest_farm_data", 
            func=get_latest_farm_data, 
            description="""
            You are a farm agent called KeSAN, created by the Grid Sphere. You can only give answer related to farm data, nothing else. Use this tool to get current farm sensor data like temperature or weather. Give the results in a very user friendly manner with emojis. Also add the Date and duration of when the data was recorded in the reply. 

            RULES:
            1 Answer as short as possible.
            2 You can only answer in two languages Hindi and English
            3 Switch your language on the basis of the user
            4 Give the date in in dd-mm-yy format
            5 Give time in am pm
            6 Never use * symbol
            """
        )
    ]
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    logger.info("LangChain Agent initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize LangChain agent: {e}")

# --- 4. Create the API Endpoint for the Chatbot ---
@app.route('/ask', methods=['POST'])
def ask_agent():
    if not agent_executor:
        return jsonify({"error": "AI agent is not available."}), 503

    user_question = request.json.get('question')
    if not user_question:
        return jsonify({"error": "No question provided."}), 400

    try:
        # Use the agent to get a response
        response = agent_executor.invoke({"input": user_question})
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


