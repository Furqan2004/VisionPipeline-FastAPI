from fastapi import FastAPI
from app.api.routes import router
from app.core.logger_utils import CustomLogger

app = FastAPI(title="Vision Pipeline API")
logger = CustomLogger("main")

# Include the API routes
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    logger.info("Vision Pipeline API starting up...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
