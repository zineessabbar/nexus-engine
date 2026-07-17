from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Api.routers import audit, system


app=FastAPI(
    title="API",
    description="Moteur d'architecture et de conformité",
    version="1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(audit.router,prefix="/api/v1/audit", tags=["Audit Workflow"])
app.include_router(system.router, prefix="/api/v1/system", tags=["System & Metrics"])