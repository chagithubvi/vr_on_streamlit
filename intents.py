from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from deepgram import Deepgram
import word2number as w2n
from zoneinfo import ZoneInfo
from datetime import datetime
import random
import re
import os
import io
import asyncio

# Load environment variables
API_KEY = os.getenv("API_KEY")
DEEPGRAM_KEY = os.getenv("DEEPGRAM_KEY")
MODEL = os.getenv("MODEL")
chat_model = ChatGroq(api_key=API_KEY, model_name=MODEL)

# Constants
SMART_HOME_KEYWORDS = ["turbine", "wind turbine", "home turbine", "power", "gear", "louvers", "battery status", "diagnostic", "speed", "standby", "lights", "fan", "ac", "air conditioner", "heater", "thermostat", "temperature", "door", "lock", "unlock", "spotlight", "humidifier", "curtain"]
SMART_HOME_ACTION = ["turn on", "shut off", "set", "check", "switch", "store", "notify", "run", "automate", "open", "adjust", "gear", "turn off", "switch on", "switch off", "set", "increase", "decrease", "open", "close", "lock", "unlock"]
ADMIN_KEYWORDS = ["wifi", "wi-fi", "new user", "family", "password", "family members", "access"]
ADMIN_ACTION = ["grant", "give", "change", "add", "remove", "show"]
GOODBYE_PHRASES = {"exit", "goodbye", "bye", "see you", "see you later", "talk to you later"}
GOODBYES = ["See you soon!", "Goodbye for now!", "Catch you later!", "Talk to you soon!", "Take care!", "Bye-bye!", "Until next time!"]
VALID_GEARS = {"1", "3", "6", "18"}
FAQ_RESPONSES = {
    "what is enercea": "Enercea is a company that builds homes and communities that generate their own clean energy, helping people live without paying utility bills.",
    "where is enercea": "Our global headquarters is in Flint City, Michigan, USA.",
    "what do they work on": "We work on BEHBs—Battery Electric Homes and Buildings—and DAKET engines for off-grid, smart, resilient communities.",
    "who built you": "Developed by Enercea to help manage your smart, sustainable living environment.",
    "who created you": "I was developed by Enercea, a clean-tech innovation company.",
    "your company": "I'm part of Enercea’s smart home ecosystem.",
    "who made you": "Enercea built me to support your battery-electric lifestyle.",
}
ADMIN_RESPONSES = [
    "That sounds like an admin-level command. Do you have the necessary access?",
    "I can’t proceed unless you’re authorized for admin operations.",
    "This action requires hoisted operations. Please authenticate first."
]

def is_smart_home_command(user_input):
    input_lower = user_input.lower()
    return any(keyword in input_lower for keyword in SMART_HOME_KEYWORDS) and any(verb in input_lower for verb in SMART_HOME_ACTION)

def get_time_by_location(user_input):
    ALIASES = {
        "california": "America/Los_Angeles", "texas": "America/Chicago", "florida": "America/New_York",
        "new york": "America/New_York", "washington": "America/Los_Angeles", "ontario": "America/Toronto",
        "british columbia": "America/Vancouver", "quebec": "America/Montreal", "mumbai": "Asia/Kolkata",
        "delhi": "Asia/Kolkata", "bangalore": "Asia/Kolkata", "maharashtra": "Asia/Kolkata",
        "gujarat": "Asia/Kolkata", "canada": "America/Toronto", "india": "Asia/Kolkata",
        "usa": "America/New_York", "united states": "America/New_York", "germany": "Europe/Berlin",
        "uk": "Europe/London", "united kingdom": "Europe/London"
    }
    match = re.search(r"\btime(?:.*in)? ([\w\s]+)", user_input.lower())
    city_guess = match.group(1).strip().lower() if match else user_input.lower().split()[-1]
    if city_guess in ALIASES:
        tz = ALIASES[city_guess]
        now = datetime.now(ZoneInfo(tz))
        city_name = tz.split("/")[-1].replace("_", " ")
        return f"The current time in {city_name.title()} is {now.strftime('%I:%M %p')}."
    for tz in ZoneInfo.available_timezones():
        if city_guess in tz.lower():
            now = datetime.now(ZoneInfo(tz))
            city_name = tz.split("/")[-1].replace("_", " ")
            return f"The current time in {city_name.title()} is {now.strftime('%I:%M %p')}."
    return None

def extract_gear_value(text):
    match = re.search(r"\bgear\s*(?:to|at)?\s*([\w\-]+)", text.lower().strip())
    if match:
        value = match.group(1).strip()
        try:
            return str(w2n.word_to_num(value))
        except:
            return value if value.isdigit() else None
    return None

def is_smart_home_question(user_input):
    input_lower = user_input.lower()
    question_indicators = ["?", "what", "status", "check", "notify", "diagnostic", "how much"]
    return any(keyword in input_lower for keyword in SMART_HOME_KEYWORDS) and any(indicator in input_lower for indicator in question_indicators)

def is_admin_command(user_input):
    input_lower = user_input.lower()
    return any(keyword in input_lower for keyword in ADMIN_KEYWORDS) and any(verb in input_lower for verb in ADMIN_ACTION)

def check_faq(user_input):
    input_lower = user_input.lower()
    return FAQ_RESPONSES.get(next((phrase for phrase in FAQ_RESPONSES if phrase in input_lower), None))

def is_continuation_of_smart_home_command(conversation_history, user_input):
    if not conversation_history:
        return False
    last_exchange = conversation_history[-1]
    return is_smart_home_command(last_exchange['user']) and not is_smart_home_command(user_input) and not is_smart_home_question(user_input)

def smart_home_response(user_input, conversation_history):
    faq_answer = check_faq(user_input)
    if faq_answer:
        return faq_answer
    history_text = "\n".join([f"User: {x['user']}\nAiva: {x['aayva']}" for x in conversation_history[-3:]])
    gear_val = extract_gear_value(user_input)
    if gear_val and gear_val in VALID_GEARS:
        system_prompt = SystemMessage(content=(
            "You are Aiva, a smart home assistant. Respond with 'Sure. Passing it to our smart home system.' or 'Of course. Passing it to our smart home system.' "
            "followed by a brief, natural comment or question. Recent conversation:\n" + history_text
        ))
        messages = [system_prompt, HumanMessage(content=user_input)]
        response = chat_model.invoke(messages, temperature=0.7, max_tokens=35)
        return response.content.strip()
    elif gear_val:
        return "Can't change, available gears are 1, 3, 6, and 18."
    if is_continuation_of_smart_home_command(conversation_history, user_input):
        return random.choice(["Sure. Passing it to our smart home system.", "Of course. Passing it to our smart home system."])
    system_prompt = SystemMessage(content=(
        "You are Aiva, a smart home assistant. Respond with 'Sure. Passing it to our smart home system.' or 'Of course. Passing it to our smart home system.' "
        "followed by a brief, natural comment or question. Recent conversation:\n" + history_text
    ))
    messages = [system_prompt, HumanMessage(content=user_input)]
    response = chat_model.invoke(messages, temperature=0.7, max_tokens=35)
    return response.content.strip()

def chat_with_Aayva(user_input, conversation_history):
    faq_answer = check_faq(user_input)
    if faq_answer:
        return faq_answer
    history_text = "\n".join([f"User: {x['user']}\nAiva: {x['aayva']}" for x in conversation_history[-3:]])
    system_prompt = SystemMessage(content=(
        "You're Aiva, a warm, helpful smart home assistant. Respond briefly and casually, like texting a friend. "
        "For goodbyes, use a warm, varied farewell. Keep responses under 30 words. Recent conversation:\n" + history_text
    ))
    messages = [system_prompt, HumanMessage(content=user_input)]
    response = chat_model.invoke(messages, temperature=0.7, max_tokens=75)
    return response.content.strip()

async def get_speech_input(audio_bytes):
    if not audio_bytes:
        return ""
    wav_buffer = io.BytesIO(audio_bytes)
    dg_client = Deepgram(DEEPGRAM_KEY)
    source = {'buffer': wav_buffer, 'mimetype': 'audio/wav'}
    try:
        response = await dg_client.transcription.prerecorded(
            source, {'model': 'nova', 'punctuate': True, 'smart_format': True}
        )
        return response['results']['channels'][0]['alternatives'][0]['transcript'].strip()
    except Exception:
        return ""

def aayva_response_from_text(user_input, conversation_history):
    if user_input.lower() in GOODBYE_PHRASES:
        response = random.choice(GOODBYES)
    elif "time" in user_input.lower():
        response = get_time_by_location(user_input) or "Hmm, I couldn’t find the current time for that location."
    elif is_smart_home_command(user_input) or is_continuation_of_smart_home_command(conversation_history, user_input):
        response = smart_home_response(user_input, conversation_history)
    elif is_smart_home_question(user_input):
        response = chat_with_Aayva(user_input, conversation_history)
    elif is_admin_command(user_input):
        response = random.choice(ADMIN_RESPONSES)
    else:
        response = chat_with_Aayva(user_input, conversation_history)
    conversation_history.append({"user": user_input, "aayva": response})
    return response, conversation_history