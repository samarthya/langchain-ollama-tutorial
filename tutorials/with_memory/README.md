# Tutorial: Add Memory to the Chain

This short tutorial shows how to add simple conversational memory to a LangChain chain so the model can retain context across invocations.

We put these files in a separate orphan Git branch called `memory-tutorial` so you can treat it as an independent tutorial without creating a new repository.

## Goal

Keep a short conversation history (chat-like) between separate `chain.invoke(...)` calls so the model can reference earlier turns.

## Approach

- Use a memory abstraction (e.g., `ConversationBufferMemory`) to collect previous inputs/responses.
- Attach the memory to an `LLMChain` (or appropriate chain abstraction) so the prompt can include the history.

## Example (conceptual, adjust imports to your LangChain version)

This example shows the concept using `LLMChain` and `ConversationBufferMemory`. Depending on your installed LangChain package versions, import paths may vary slightly.

```python
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_ollama.chat_models import ChatOllama

def demo_memory_flow():
    # load prompt template (same as in main.py)
    prompt_text = open("prompts/summary_template.txt").read()

    prompt = PromptTemplate(input_variables=["information"], template=prompt_text)

    # initialize the Ollama LLM (reuse your init_ollama_llm helper)
    llm = ChatOllama(temperature=0.0, model="gpt-oss:20b", base_url="http://localhost:11434")

    # create a conversation memory buffer
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # create an LLMChain with memory attached
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    # First invocation: system/user provides information
    resp1 = chain.run({"information": "Alice is a software engineer who loves cats."})
    print("First response:\n", resp1)

    # Second invocation: supply new info; chain includes chat_history in the prompt
    resp2 = chain.run({"information": "She recently started a new job at a robotics startup."})
    print("Second response (with memory):\n", resp2)

    # The memory object now contains the conversation and can be inspected
    print("Memory contents:\n", memory.load_memory_variables({}))

if __name__ == "__main__":
    demo_memory_flow()
```

Notes
- This code is intentionally simple and conceptual. Your installed LangChain version may use a slightly different API (for example, some versions expect `chain.invoke()` or `chain.run()` with different argument shapes).
- `ConversationBufferMemory` stores previous messages and exposes them via `memory.load_memory_variables({})`. The prompt template can reference the memory via the `memory_key` you supplied (in the example, `chat_history`).

## How to use this branch

Locally, to switch to the orphan branch (created for you):

```bash
git fetch origin
git checkout memory-tutorial
```

This branch contains only the tutorial files under `tutorials/with_memory/` so you can experiment without affecting the main branch.
