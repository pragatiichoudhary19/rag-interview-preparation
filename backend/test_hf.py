
import asyncio

from app.services.llm_service import LLMService, msgs


async def test_llama():
    print("Connecting to Ollama...")

    llm = LLMService()

    try:
        response = await llm.ainvoke_text(
            msgs(
                "You are a helpful AI interview preparation assistant.",
                "Explain Python in simple words with one example."
            )
        )

        print("\nLlama 3.1 Response:")
        print(response)

        print("\nSUCCESS: Ollama is working!")

    except Exception as e:
        print("\nERROR:", str(e))


if __name__ == "__main__":
    asyncio.run(test_llama())

    asyncio.run(test_llama())
