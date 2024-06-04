#!/usr/bin/env python3
import sys
import traceback
import signal
import os
import torch
import logging
from torch.cuda.amp import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig, Conversation
import json

# Load configuration from config.json
with open('src/config.json', 'r') as config_file:
    config = json.load(config_file)

LOG_FILE = config["LOG_FILE"]
LOGGING_FILE = config["LOGGING_FILE"]
logging_level = config["logging_level"]
logging_format = config.get("logging_format", '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
api_token = config.get("api_token")

def save_context(context):
    with open(LOG_FILE, "w") as f:
        f.writelines(context)

def load_context():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return f.readlines()
    return []

def signal_handler(sig, frame):
    save_context(context)
    sys.exit(2)

signal.signal(signal.SIGINT, signal_handler)

def create_pipeline(task, model, tokenizer):
    try:
        return pipeline(task, model=model, tokenizer=tokenizer)
    except Exception as e:
        logging.warning(f"{model.config.model_type} does not use Task: '{task}': {e}")
        return None

def main():
    global context
    logging.basicConfig(filename=LOGGING_FILE, level=logging_level, format=logging_format)
    try:
        print("Welcome! Model loading....")
        hub_token = api_token or os.environ.get("HF_API_TOKEN")
        if not hub_token:
            raise ValueError("Hugging Face API token not found in environment variables.")

        model_dir = config['win32_model_dir'] if sys.platform == 'win32' else config['wsl_model_dir']
        if not os.path.exists(model_dir):
            model_dir = config['model']

        logging.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        logging.info("Loading complete!") 
        
        if device == 'cuda' and torch.cuda.is_available():
            model = model.to('cuda:0')
            logging.info("Model moved to GPU")
        else:
            model = model.to('cpu')
            logging.info("Model moved to CPU")

        config = AutoConfig.from_pretrained(model_dir)
        tasks = getattr(config, 'supported_tasks', [
            "feature-extraction", "text-classification", "sentiment-analysis", 
            "question-answering", "summarization", "translation", 
            "text-generation", "conversational"
        ])
        logging.info(f"Model supports tasks: {tasks}")
        
        pipelines = {task: create_pipeline(task, model, tokenizer) for task in tasks}
        context = load_context() if os.path.exists(LOG_FILE) else []

        while True:
            input_text = input('Me:  ').strip()
            if input_text.lower() == 'quit':
                save_context(context)
                break
            context.append(f"User: {input_text}\n")
            
            logging.info("Bot(s):")
            responses = ["Bot(s):"]
            with autocast():
                for task, pipe in pipelines.items():
                    if pipe:
                        try:
                            if task == "question-answering":
                                
                                output = pipe(inputs=input_text, context=" ".join(context))
                            elif task == "conversational":
                                conversation = "\n".join(input_text,Conversation("\n".join(context)))
                                output = pipe(conversation)
                                conversation.add_ai_response(f"ChattyBot: {output}")
                                print(f"ChattyBot: {output}")
                            else:
                                output = pipe(input_text)
                            responses.append(f"{task.capitalize()}Bot: {output}")
                            logging.info(f"{task.capitalize()}Bot: {output}")
                        except Exception as e:
                            logging.error(f"Error in {task} pipeline: {e}")
            context.append("\n".join(responses))
    except Exception as e:
        logging.error(f"Error: {e}")
        traceback.print_exc()
    finally:
        save_context(context)

if __name__ == "__main__":
    main()
