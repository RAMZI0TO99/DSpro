from app.main import app

if __name__ == "__main__":
    import uvicorn
    import asyncio
    import sys

    # Silences the WinError 10054 connection drops on Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run(app, host="0.0.0.0", port=8000)