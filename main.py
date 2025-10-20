import os

from dotenv import load_dotenv
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.llms import OllamaLLM

load_dotenv()


def main():
    print(f" {os.environ.get('OLLAMA_HOST')}:{os.environ.get('OLLAMA_PORT')}")
    print(f" Hello World!")


if __name__ == "__main__":
    main()
