
import asyncio

from app.database import mongo
from app.config import get_settings


async def test_connection():
    settings = get_settings()

    print("Database:", settings.mongodb_db_name)
    print("Connecting to MongoDB Atlas...")

    try:
        db = mongo.get_db()
        await db.command("ping")

        print("SUCCESS: MongoDB Atlas connected!")
        print("Database connection is working.")

    except Exception as e:
        print("CONNECTION FAILED:", str(e))

    finally:
        await mongo.close()


if __name__ == "__main__":
    asyncio.run(test_connection())
