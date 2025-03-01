import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import gradio as gr
import requests
from contextlib import contextmanager
from datetime import datetime
import uuid
import logging
import speech_recognition as sr
from moviepy import VideoFileClip

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app and SQLAlchemy
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///conversations.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Set Gemini API key
gemini_api_key = ""  # Replace with your actual API key
gemini_api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Database models
class Character(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    description = db.Column(db.Text, nullable=False)
    prompt_template = db.Column(db.Text, nullable=False)

class Conversation(db.Model):
    __tablename__ = 'conversation'
    
    id = db.Column(db.Integer, primary_key=True)
    character_id = db.Column(db.Integer, db.ForeignKey('character.id'), nullable=False)
    user_input = db.Column(db.Text, nullable=True)
    bot_response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    chat_id = db.Column(db.String(36), nullable=True)
    user_id = db.Column(db.Integer, nullable=False)

@contextmanager
def app_context():
    with app.app_context():
        yield

def add_predefined_characters():
    with app_context():
        characters = [
            {
                "name": "Chuck the Clown",
                "description": "A funny clown who tells jokes and entertains.",
                "prompt_template": "You are Chuck the Clown, always ready with a joke and entertainment. Be upbeat, silly, and include jokes in your responses."
            },
            {
                "name": "Sarcastic Pirate",
                "description": "A pirate with a sharp tongue and a love for treasure.",
                "prompt_template": "You are a Sarcastic Pirate, ready to share your tales of adventure. Use pirate slang, be witty, sarcastic, and mention your love for treasure and the sea."
            },
            {
                "name": "Professor Sage",
                "description": "A wise professor knowledgeable about many subjects.",
                "prompt_template": "You are Professor Sage, sharing wisdom and knowledge. Be scholarly, thoughtful, and provide educational information in your responses."
            }
        ]

        for char_data in characters:
            if not Character.query.filter_by(name=char_data["name"]).first():
                new_character = Character(
                    name=char_data["name"],
                    description=char_data["description"],
                    prompt_template=char_data["prompt_template"]
                )
                db.session.add(new_character)
                logger.info(f"Adding predefined character: {char_data['name']}")
        
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error adding predefined characters: {e}")

def add_character(name, description, prompt_template):
    with app_context():
        try:
            if Character.query.filter_by(name=name).first():
                return f"Character '{name}' already exists!"

            new_character = Character(
                name=name,
                description=description,
                prompt_template=prompt_template
            )
            db.session.add(new_character)
            db.session.commit()
            logger.info(f"Successfully added character: {name}")
            return f"Character '{name}' added successfully!\nDescription: {description}"
        
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error adding character: {e}")
            return f"An error occurred while adding the character: {str(e)}"

def get_existing_characters():
    with app_context():
        try:
            characters = Character.query.all()
            return [(char.name, char.description) for char in characters]
        except Exception as e:
            logger.error(f"Error retrieving characters: {e}")
            return [("Error retrieving characters", str(e))]

def chat_with_character(character_name, user_input, user_id, chat_id=None):
    with app_context():
        try:
            character = Character.query.filter_by(name=character_name).first()
            
            if not character:
                return "Character not found.", None
            
            if not chat_id:
                chat_id = str(uuid.uuid4())
            
            # Get previous conversations for this chat_id if it exists
            previous_conversations = []
            if chat_id:
                previous_conversations = Conversation.query.filter_by(
                    user_id=user_id, 
                    chat_id=chat_id
                ).order_by(Conversation.timestamp).all()
            else:
                # If no chat_id, get recent conversations for this user with this character
                previous_conversations = Conversation.query.filter_by(
                    user_id=user_id,
                    character_id=character.id
                ).order_by(Conversation.timestamp.desc()).limit(10).all()
                previous_conversations.reverse()  # Most recent last
            
            context_prompt = " ".join([f"User: {conv.user_input}\nBot: {conv.bot_response}" for conv in previous_conversations])
            prompt_template = character.prompt_template
            full_prompt = f"{prompt_template}\n{context_prompt}\nUser: {user_input}\nBot:"

            payload = {
                "contents": [{
                    "parts": [{"text": full_prompt}]
                }]
            }

            headers = {
                'Content-Type': 'application/json'
            }

            response = requests.post(
                gemini_api_url,
                headers=headers,
                json=payload,
                params={'key': gemini_api_key}
            )

            if response.status_code == 200:
                response_data = response.json()
                if 'candidates' in response_data and len(response_data['candidates']) > 0:
                    bot_response = response_data['candidates'][0]['content']['parts'][0]['text']
                    
                    conversation = Conversation(
                        character_id=character.id,
                        user_input=user_input,
                        bot_response=bot_response,
                        chat_id=chat_id,
                        user_id=user_id
                    )
                    db.session.add(conversation)
                    db.session.commit()
                    logger.info(f"Saved conversation with chat_id: {chat_id}")
                    return bot_response, chat_id
                else:
                    return "An error occurred while generating content: Unexpected response format.", chat_id
            else:
                logger.error(f"Error from Gemini API: {response.json()}")
                return f"An error occurred while generating content: {response.status_code} - {response.text}", chat_id

        except Exception as e:
            logger.error(f"Unexpected error in chat_with_character: {e}")
            return f"An unexpected error occurred: {str(e)}", chat_id

def speech_to_text(audio_file):
    """Convert audio file to text using SpeechRecognition."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            logger.info(f"Transcribed text: {text}")
            return text
        except sr.UnknownValueError:
            logger.error("Could not understand audio")
            return "Could not understand audio"
        except sr.RequestError as e:
            logger.error(f"Could not request results from Google Speech Recognition service; {e}")
            return f"Error: {str(e)}"

def extract_audio_from_video(video_file):
    """Extract audio from video and return the audio file path."""
    audio_file_path = "temp_audio.wav"  # Temporary file name
    try:
        with VideoFileClip(video_file) as video:
            video.audio.write_audiofile(audio_file_path)
        logger.info(f"Extracted audio to {audio_file_path}")
        return audio_file_path
    except Exception as e:
        logger.error(f"Error extracting audio from video: {e}")
        return None  # Return None if there's an error

def process_video(video_file):
    """Process video file to extract text."""
    audio_file = extract_audio_from_video(video_file)
    if audio_file:
        text = speech_to_text(audio_file)
        # Clean up the temporary file
        try:
            os.remove(audio_file)
        except:
            pass
        return text
    return "Failed to process video"

def get_chat_history(user_id):
    """Retrieve chat history for a specific user ID."""
    with app_context():
        # Get unique chat sessions
        chat_sessions = db.session.query(Conversation.chat_id, Character.name, 
                                        db.func.min(Conversation.timestamp).label('start_time'))\
            .join(Character, Conversation.character_id == Character.id)\
            .filter(Conversation.user_id == user_id)\
            .group_by(Conversation.chat_id)\
            .order_by(db.text('start_time DESC'))\
            .all()
        
        result = []
        for chat_id, character_name, start_time in chat_sessions:
            # Get the first message from each conversation to use as a title
            first_message = Conversation.query.filter_by(
                chat_id=chat_id, 
                user_id=user_id
            ).order_by(Conversation.timestamp).first()
            
            if first_message:
                # Calculate message count
                message_count = Conversation.query.filter_by(
                    chat_id=chat_id,
                    user_id=user_id
                ).count()
                
                formatted_date = start_time.strftime("%Y-%m-%d %H:%M:%S")
                preview = first_message.user_input[:30] + "..." if len(first_message.user_input) > 30 else first_message.user_input
                result.append((chat_id, character_name, preview, formatted_date, message_count))
                
        return result

def get_chat_messages(chat_id, user_id):
    """Get all messages for a specific chat session."""
    with app_context():
        messages = Conversation.query.filter_by(
            chat_id=chat_id, 
            user_id=user_id
        ).order_by(Conversation.timestamp).all()
        
        return [(msg.user_input, msg.bot_response) for msg in messages]

def auto_select_character(user_input):
    user_input = user_input.lower()
    
    education_keywords = ["learn", "study", "education", "knowledge", "science", "history", "math", "theory", "research"]
    entertainment_keywords = ["joke", "funny", "laugh", "entertain", "comedy", "silly"]
    adventure_keywords = ["adventure", "sea", "pirate", "treasure", "sail", "voyage"]
    
    if any(keyword in user_input for keyword in education_keywords):
        return "Professor Sage"
    elif any(keyword in user_input for keyword in entertainment_keywords):
        return "Chuck the Clown"
    elif any(keyword in user_input for keyword in adventure_keywords):
        return "Sarcastic Pirate"
    
    return None

def create_interface():
    with app.app_context():
        add_predefined_characters()  # Add predefined characters if needed
    
    with gr.Blocks(title="Character Chat System", theme=gr.themes.Base(), css="""
        :root {
            --main-color: #4A90E2;
            --accent-color: #FF6B6B;
            --bg-color: #1a1a2e;
            --text-color: #f1f1f1;
            --card-bg: #16213e;
            --border-color: #0f3460;
        }
        
        body {
            background: var(--bg-color);
            color: var(--text-color);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            transition: all 0.3s ease;
        }
        
        /* Animated background */
        body::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(125deg, #1a1a2e 0%, #16213e 30%, #0f3460 70%, #1a1a2e 100%);
            background-size: 400% 400%;
            z-index: -1;
            animation: gradient 15s ease infinite;
        }
        
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Floating particles */
        .particle {
            position: fixed;
            border-radius: 50%;
            opacity: 0.3;
            pointer-events: none;
            z-index: -1;
            animation: float 20s infinite linear;
        }
        
        @keyframes float {
            0% { transform: translateY(0) rotate(0deg); }
            100% { transform: translateY(-100vh) rotate(360deg); }
        }
        
        /* Generate 20 particles */
        .gradio-container::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
        }
        
        #title {
            text-align: center;
            font-size: 3.5em;
            font-weight: 700;
            background: linear-gradient(90deg, var(--main-color), var(--accent-color), var(--main-color));
            background-size: 200% auto;
            color: transparent;
            -webkit-background-clip: text;
            background-clip: text;
            animation: shine 3s linear infinite;
            margin-bottom: 30px;
            text-shadow: 0 0 10px rgba(74, 144, 226, 0.3);
            letter-spacing: 1px;
        }
        
        @keyframes shine {
            to {
                background-position: 200% center;
            }
        }
        
        .gradio-container {
            max-width: 90% !important;
            margin: 20px auto !important;
            border-radius: 15px !important;
            background: rgba(22, 33, 62, 0.8) !important;
            backdrop-filter: blur(10px) !important;
            border: 1px solid var(--border-color) !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
            padding: 30px !important;
            overflow: hidden !important;
            position: relative !important;
        }
        
        .gradio-container::after {
            content: "";
            position: absolute;
            inset: 0;
            border-radius: 15px;
            padding: 2px;
            background: linear-gradient(45deg, var(--main-color), var(--accent-color));
            mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            pointer-events: none;
            z-index: -1;
        }
        
        /* Tabs styling */
        .tabs {
            background: var(--card-bg) !important;
            border-radius: 10px !important;
            overflow: hidden !important;
            margin-bottom: 20px !important;
        }
        
        .tab-nav {
            background: var(--card-bg) !important;
            border-bottom: 1px solid var(--border-color) !important;
        }
        
        .tab-nav button {
            color: var(--text-color) !important;
            font-weight: 600 !important;
            padding: 12px 20px !important;
            border-radius: 0 !important;
            transition: all 0.3s !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        .tab-nav button::before {
            content: "";
            position: absolute;
            bottom: 0;
            left: 50%;
            width: 0;
            height: 3px;
            background: linear-gradient(to right, var(--main-color), var(--accent-color));
            transform: translateX(-50%);
            transition: width 0.3s;
        }
        
        .tab-nav button:hover::before,
        .tab-nav button.selected::before {
            width: 80%;
        }
        
        .tab-nav button.selected {
            color: white !important;
            background: transparent !important;
        }
        
        /* Form elements styling */
        input[type="text"], 
        input[type="password"], 
        textarea, 
        select,
        .gr-input,
        .gr-box,
        .gr-padded {
            background: rgba(15, 25, 50, 0.7) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
            padding: 12px !important;
            color: var(--text-color) !important;
            transition: all 0.3s !important;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2) !important;
        }
        
        input[type="text"]:focus, 
        input[type="password"]:focus, 
        textarea:focus, 
        select:focus,
        .gr-input:focus {
            border-color: var(--main-color) !important;
            box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.3) !important;
            outline: none !important;
        }
        
        /* Button styling */
        .gr-button,
        button {
            background: linear-gradient(45deg, var(--main-color), #357ABD) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 12px 24px !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(74, 144, 226, 0.3) !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        .gr-button:hover,
        button:hover {
            background: linear-gradient(45deg, #357ABD, var(--main-color)) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(74, 144, 226, 0.4) !important;
        }
        
        .gr-button:active,
        button:active {
            transform: translateY(1px) !important;
            box-shadow: 0 2px 10px rgba(74, 144, 226, 0.3) !important;
        }
        
        .gr-button::before,
        button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: all 0.4s;
        }
        
        .gr-button:hover::before,
        button:hover::before {
            left: 100%;
        }
        
        /* Primary button */
        .gr-button.gr-button-primary {
            background: linear-gradient(45deg, var(--accent-color), #E85D5D) !important;
            box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3) !important;
        }
        
        .gr-button.gr-button-primary:hover {
            background: linear-gradient(45deg, #E85D5D, var(--accent-color)) !important;
            box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4) !important;
        }
        
        /* Chatbot styling */
        .gr-chatbot {
            background: var(--card-bg) !important;
            border-radius: 12px !important;
            border: 1px solid var(--border-color) !important;
            padding: 0 !important;
            min-height: 400px !important;
            max-height: 600px !important;
            overflow-y: auto !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
            margin-bottom: 20px !important;
        }
        
        .gr-chatbot .message {
            padding: 12px 16px !important;
            margin: 8px !important;
            border-radius: 10px !important;
            position: relative !important;
            max-width: 80% !important;
            animation: message-fade-in 0.3s ease !important;
        }
        
        @keyframes message-fade-in {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .gr-chatbot .user-message {
            background: linear-gradient(135deg, var(--main-color), #357ABD) !important;
            color: white !important;
            border-radius: 12px 12px 0 12px !important;
            align-self: flex-end !important;
            margin-left: auto !important;
            box-shadow: 0 2px 10px rgba(74, 144, 226, 0.3) !important;
        }
        
        .gr-chatbot .bot-message {
            background: var(--card-bg) !important;
            color: var(--text-color) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px 12px 12px 0 !important;
            margin-right: auto !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Chat typing indicator */
        .typing-indicator {
            display: inline-block;
            padding: 6px 12px;
            background: var(--card-bg);
            border-radius: 20px;
            margin: 10px;
        }
        
        .typing-indicator span {
            height: 10px;
            width: 10px;
            float: left;
            margin: 0 1px;
            background-color: #9E9EA1;
            display: block;
            border-radius: 50%;
            opacity: 0.4;
        }
        
        .typing-indicator span:nth-of-type(1) {
            animation: typing 1s infinite 0s;
        }
        
        .typing-indicator span:nth-of-type(2) {
            animation: typing 1s infinite 0.2s;
        }
        
        .typing-indicator span:nth-of-type(3) {
            animation: typing 1s infinite 0.4s;
        }
        
        @keyframes typing {
            0% { transform: translateY(0px); }
            33% { transform: translateY(-5px); }
            66% { transform: translateY(0px); }
        }
        
        /* Dataframe styling */
        table.dataframe {
            width: 100% !important;
            border-collapse: separate !important;
            border-spacing: 0 !important;
            border-radius: 10px !important;
            overflow: hidden !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
            margin: 20px 0 !important;
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
        }
        
        table.dataframe th {
            background: rgba(74, 144, 226, 0.2) !important;
            color: var(--text-color) !important;
            padding: 12px 15px !important;
            font-weight: 600 !important;
            text-align: left !important;
            border-bottom: 2px solid var(--border-color) !important;
            position: relative !important;
        }
        
        table.dataframe td {
            padding: 12px 15px !important;
            border-bottom: 1px solid var(--border-color) !important;
            color: var(--text-color) !important;
            transition: all 0.2s !important;
        }
        
        table.dataframe tr:hover td {
            background: rgba(74, 144, 226, 0.05) !important;
        }
        
        table.dataframe tr:last-child td {
            border-bottom: none !important;
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--card-bg);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 10px;
            transition: all 0.3s;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--main-color);
        }
        
        /* Chat history cards */
        .chat-history-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            cursor: pointer;
            transition: all 0.3s;
            position: relative;
            overflow: hidden;
        }
        
        .chat-history-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
            border-color: var(--main-color);
        }
        
        .chat-history-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 5px;
            height: 100%;
            background: linear-gradient(to bottom, var(--main-color), var(--accent-color));
        }
        
        .chat-history-card h3 {
            margin-top: 0;
            color: white;
            font-size: 1.2em;
        }
        
        .chat-history-card p {
            margin: 5px 0;
            color: #ccc;
        }
        
        .chat-history-badge {
            position: absolute;
            top: 10px;
            right: 10px;
            background: var(--main-color);
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
        }
        
        /* Audio/Video recording styles */
        .media-recorder {
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 10px !important;
            padding: 15px !important;
            margin-bottom: 15px !important;
        }
        
        .media-recorder .recording-indicator {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--accent-color);
            display: inline-block;
            margin-right: 10px;
            animation: pulse-recording 1.5s infinite;
        }
        
        @keyframes pulse-recording {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.2); opacity: 0.5; }
            100% { transform: scale(1); opacity: 1; }
        }
        
        /* Animated emoji */
        .blinking-emoji {
            display: inline-block;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.2); }
            100% { transform: scale(1); }
        }
        
        /* Loading animation */
        .loading-animation {
            width: 60px;
            height: 60px;
            margin: 20px auto;
            position: relative;
        }
        
        .loading-animation div {
            position: absolute;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: var(--main-color);
            animation: loading-animation 1.2s linear infinite;
        }
        
        .loading-animation div:nth-child(1) {
            top: 8px;
            left: 8px;
            animation-delay: 0s;
        }
        
        .loading-animation div:nth-child(2) {
            top: 8px;
            left: 32px;
            animation-delay: -0.4s;
        }
        
        .loading-animation div:nth-child(3) {
            top: 32px;
            left: 8px;
            animation-delay: -0.8s;
        }
        
        .loading-animation div:nth-child(4) {
            top: 32px;
            left: 32px;
            animation-delay: -0.4s;
        }
        
        @keyframes loading-animation {
            0%, 100% {
                opacity: 1;
                transform: scale(1);
            }
            50% {
                opacity: 0.5;
                transform: scale(0.5);
            }
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .gradio-container {
                max-width: 95% !important;
                padding: 15px !important;
            }
            
            #title {
                font-size: 2.5em !important;
            }
            
            .gr-chatbot {
                min-height: 300px !important;
            }
            
            .gr-chatbot .message {
                max-width: 90% !important;
            }
        }
        
        /* Create dynamic background particles */
        .gradio-container::after {
            content: "";
        }
        
        /* JavaScript to add dynamic elements */
        <script>
            // Add floating particles
            function createParticles() {
                const container = document.querySelector('.gradio-container');
                const particleCount = 30;
                
                for (let i = 0; i < particleCount; i++) {
                    const particle = document.createElement('div');
                    particle.classList.add('particle');
                    
                    // Random properties
                    const size = Math.random() * 5 + 2;
                    const posX = Math.random() * 100;
                    const posY = Math.random() * 100;
                    const delay = Math.random() * 10;
                    const duration = Math.random() * 10 + 10;
                    
                    // Set styles
                    particle.style.width = `${size}px`;
                    particle.style.height = `${size}px`;
                    particle.style.left = `${posX}%`;
                    particle.style.bottom = `${posY}%`;
                    particle.style.animationDelay = `${delay}s`;
                    particle.style.animationDuration = `${duration}s`;
                    particle.style.background = i % 2 === 0 ? 
                        `rgba(74, 144, 226, ${Math.random() * 0.5 + 0.1})` : 
                        `rgba(255, 107, 107, ${Math.random() * 0.5 + 0.1})`;
                    
                    // Add to container
                    container.appendChild(particle);
                }
            }
            
            // Make chat history items clickable
            function setupChatHistoryEvents() {
                setTimeout(() => {
                    const chatHistoryItems = document.querySelectorAll('.chat-history-card');
                    chatHistoryItems.forEach(item => {
                        item.addEventListener('click', () => {
                            // Get the chat ID from the data attribute
                            const chatId = item.getAttribute('data-chat-id');
                            // Trigger the click on the hidden button with this chat ID
                            document.querySelector(`button[data-chat-id="${chatId}"]`).click();
                        });
                    });
                }, 1000);
            }
            
            // Call functions when DOM is loaded
            document.addEventListener('DOMContentLoaded', () => {
                createParticles();
                setupChatHistoryEvents();
            });
        </script>
    """) as iface:
        current_chat_id = gr.State(value=None)  # State to track the current chat ID
        user_id = gr.State(value=None)  # State to track user ID
        chat_messages = gr.State(value=[])  # State to store chat messages
        
        gr.Markdown(
            "# 🎭 Character Chat System 🎭",
            elem_id="title"
        )
        
        # Add HTML for animated background elements
        gr.HTML("""
        <div class="dynamic-background">
            <div class="loading-animation">
                <div></div>
                <div></div>
                <div></div>
                <div></div>
            </div>
        </div>
        """)
        
        with gr.Tab("Sign In"):
            user_id_input = gr.Textbox(label="User ID", placeholder="Enter your User ID", interactive=True, lines=2)
            sign_in_btn = gr.Button("Sign In", variant="primary")
            sign_in_response = gr.Textbox(label="Sign In Response", interactive=False)

            def sign_in(user_id_input):
                user_id.value = user_id_input
                load_history = load_chat_history(user_id_input)
                return f"Welcome, {user_id_input}!", user_id_input
            
            sign_in_btn.click(fn=sign_in, inputs=[user_id_input], outputs=[sign_in_response, user_id])

        with gr.Tab("Admin: Add Character"):
            with gr.Row():
                name_input = gr.Textbox(label="Character Name", placeholder="Enter character name")
                description_input = gr.Textbox(label="Character Description", placeholder="Enter character description")
                prompt_input = gr.Textbox(label="Prompt Template", placeholder="Enter character prompt template", lines=3)
                add_character_btn = gr.Button("Add Character", variant="primary")
                add_character_response = gr.Textbox(label="Response", interactive=False)

                add_character_btn.click(
                    fn=add_character,
                    inputs=[name_input, description_input, prompt_input],
                    outputs=[add_character_response]
                )
                
                character_list = gr.Dataframe(
                    value=get_existing_characters(),
                    headers=["Name", "Description"],
                    interactive=False
                )
                
                refresh_characters_btn = gr.Button("Refresh Character List")
                
                def refresh_characters():
                    return gr.update(value=get_existing_characters())
                
                refresh_characters_btn.click(fn=refresh_characters, outputs=[character_list])
        
        with gr.Tab("Chat with Character"):
            with gr.Row():
                with gr.Column(scale=2):
                    character_dropdown = gr.Dropdown(
                        label="Choose Character", 
                        choices=[char[0] for char in get_existing_characters()],
                        interactive=True
                    )
                    user_input = gr.Textbox(label="Your Message", placeholder="Type your message or use audio input", lines=2)
                    
                    with gr.Row():
                        auto_select_btn = gr.Button("Auto Select Character")
                        send_btn = gr.Button("Send", variant="primary")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            audio_input = gr.Audio( type="filepath", label="Audio Input")
                        
                        with gr.Column(scale=1):
                            video_input = gr.Video( label="Video Input")
                    
                    chat_display = gr.Chatbot(label="Chat Responses", height=400)
                    
                    # Hidden button to load a chat
                    load_chat_btn = gr.Button("Load Chat", visible=False)
                    chat_id_input = gr.Textbox(visible=False)

            def auto_select(character_name, user_input):
                selected_character = auto_select_character(user_input)
                if selected_character:
                    return selected_character
                return character_name
            
            def transcribe_audio(audio_file, text_input):
                if audio_file is None:
                    return text_input
                
                transcribed_text = speech_to_text(audio_file)
                if transcribed_text and transcribed_text != "Could not understand audio":
                    return transcribed_text
                return text_input
            
            def transcribe_video(video_file, text_input):
                if video_file is None:
                    return text_input
                
                transcribed_text = process_video(video_file)
                if transcribed_text and transcribed_text != "Failed to process video":
                    return transcribed_text
                return text_input
            
            def handle_chat(character_name, user_input, user_id_val, current_chat_id_val=None):
                if not user_id_val:
                    return [(None, "Please sign in first!")], None
                
                if not user_input or user_input.strip() == "":
                    return [(None, "Please enter a message!")], current_chat_id_val
                
                response, new_chat_id = chat_with_character(character_name, user_input, user_id_val, current_chat_id_val)
                
                # Update our chat ID if this is a new conversation
                if current_chat_id_val is None:
                    current_chat_id_val = new_chat_id
                
                return [(user_input, response)], current_chat_id_val
            
            def load_existing_chat(chat_id, user_id_val):
                if not user_id_val or not chat_id:
                    return [], None
                
                messages = get_chat_messages(chat_id, user_id_val)
                return messages, chat_id
            
            auto_select_btn.click(fn=auto_select, inputs=[character_dropdown, user_input], outputs=[character_dropdown])
            
            audio_input.change(fn=transcribe_audio, inputs=[audio_input, user_input], outputs=[user_input])
            video_input.change(fn=transcribe_video, inputs=[video_input, user_input], outputs=[user_input])
            
            send_btn.click(
                fn=handle_chat, 
                inputs=[character_dropdown, user_input, user_id, current_chat_id], 
                outputs=[chat_display, current_chat_id]
            )
            
            load_chat_btn.click(
                fn=load_existing_chat,
                inputs=[chat_id_input, user_id],
                outputs=[chat_display, current_chat_id]
            )
        
        with gr.Tab("Chat History"):
            history_container = gr.HTML(load_chat_history(user_id.value) if user_id.value else "Please sign in to view your chat history.")
            
            def format_chat_history(history_data):
                if not history_data:
                    return "<div class='empty-history'>No chat history available.</div>"
                
                html = "<div class='chat-history-container'>"
                for chat_id, character, preview, date, msg_count in history_data:
                    html += f"""
                    <div class='chat-history-card' data-chat-id='{chat_id}'>
                        <div class='chat-history-badge'>{msg_count} messages</div>
                        <h3>{character}</h3>
                        <p><strong>Started:</strong> {date}</p>
                        <p><strong>First message:</strong> {preview}</p>
                    </div>
                    <button style='display:none;' data-chat-id='{chat_id}' id='load-chat-{chat_id}'></button>
                    """
                html += "</div>"
                return html
            
            def load_chat_history(user_id_val):
                if not user_id_val:
                    return "<div class='error-message'>Please sign in first to view your chat history.</div>"
                
                history = get_chat_history(user_id_val)
                return format_chat_history(history)
            
            view_history_btn = gr.Button("View History", variant="primary")
            view_history_btn.click(fn=load_chat_history, inputs=[user_id], outputs=[history_container])

            # Create hidden buttons for each chat history item
            def setup_chat_history_buttons(history_container):
                # This function will be executed by the JavaScript
                return history_container
            
            history_container.change(fn=setup_chat_history_buttons, inputs=[history_container], outputs=[history_container])

    return iface

if __name__ == "__main__":
    with app.app_context():
        #db.drop_all()
        db.create_all()  # Ensure tables are created
        add_predefined_characters()  # Add predefined characters if needed
    
    chat_interface = create_interface()
    logger.info("Starting Gradio interface...")
    chat_interface.launch(share=True)
