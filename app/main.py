from fastapi import FastAPI
from app.api.routes import router

app = FastAPI()

# Include the router
app.include_router(router) 