import os

from dotenv import load_dotenv
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate


load_dotenv()

information = """
Elon must the millionaire and the geek who rules the technology space. He is the owner of Tesla, SpaceX and Twitter or X.
Elon Reeve Musk (born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, Twitter, and xAI. Musk has been the wealthiest person in the world since 2021; as of October 2025, Forbes estimates his net worth to be US$500 billion.
Born into a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he had obtained Canadian citizenship at birth through his Canadian-born mother. He received bachelor's degrees in 1997 from the University of Pennsylvania in Philadelphia, United States, before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. That year, Musk also became an American citizen.
In 2002, Musk founded the space technology company SpaceX, becoming its CEO and chief engineer; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined the automaker Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence (AI) research, but later left; growing discontent with the organization's direction and their leadership in the AI boom in the 2020s led him to establish xAI. In 2022, he acquired the social network Twitter, implementing significant changes, and rebranding it as X in 2023. His other businesses include the neurotechnology company Neuralink, which he co-founded in 2016, and the tunneling company the Boring Company, which he founded in 2017.
Musk was the largest donor in the 2024 U.S. presidential election, and is a supporter of global far-right figures, causes, and political parties. In early 2025, he served as senior advisor to United States president Donald Trump and as the de facto head of DOGE. After a public feud with Trump, Musk left the Trump administration and returned to his technology companies.
Musk's political activities, views, and statements have made him a polarizing figure, especially following the COVID-19 pandemic. He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation and promoting conspiracy theories, and affirming antisemitic, racist, and transphobic comments. His acquisition of Twitter was controversial due to a subsequent increase in hate speech and the spread of misinformation on the service. His role in the second Trump administration attracted public backlash, particularly in response to DOGE.
"""

def load_prompt(path: str) -> str:
    """Load a prompt template from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def init_ollama_llm(model: str = "gpt-oss:20b", temperature: float = 0.0) -> ChatOllama:
    """Initialize and return a ChatOllama LLM client using OLLAMA_HOST/PORT from env."""
    host = os.environ.get("OLLAMA_HOST", "localhost")
    port = os.environ.get("OLLAMA_PORT", "11434")
    base_url = f"http://{host}:{port}"
    return ChatOllama(temperature=temperature, model=model, base_url=base_url, validate_model_on_init=True)


def build_chain(prompt_template_str: str, input_variables: list[str], llm: ChatOllama):
    """Create a PromptTemplate and compose it with the LLM to return a chain.

    Returns a chain object that can be invoked with the required inputs.
    """
    prompt = PromptTemplate(input_variables=input_variables, template=prompt_template_str)
    return prompt | llm


def main() -> None:
    host = os.environ.get("OLLAMA_HOST", "localhost")
    port = os.environ.get("OLLAMA_PORT", "11434")
    print(f"Using Ollama server at {host}:{port}")

    prompt_text = load_prompt("prompts/summary_template.txt")

    llm = init_ollama_llm()
    chain = build_chain(prompt_text, ["information"], llm)

    # Invoke the chain
    response = chain.invoke(input={"information": information})
    # Response is a message-like object; print the content for demonstration
    print("Response:\n", getattr(response, "content", str(response)))


if __name__ == "__main__":
    main()
