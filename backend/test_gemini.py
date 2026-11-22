
import asyncio
from app.services.llm_service import LLMService, msgs

async def test_gemini():
    service = LLMService()

    try:
        print("Sending request to Gemini...")

        response = await service.ainvoke_text(
            msgs(
                "You are a helpful AI assistant.",
                "Explain RAG in two simple sentences."
            )
        )

        print("\nGemini response:")
        print(response)
        print("\nSUCCESS: Gemini API is working!")

    except Exception as e:
        print("\nGEMINI TEST FAILED:")
        print(type(e).__name__, str(e))

if __name__ == "__main__":
    asyncio.run(test_gemini())
