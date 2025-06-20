from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import logging
from rag_pipeline import generate_question  # Updated import
from res_match import match_resume_to_jd # Updated import
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# No need to initialize RAGPipeline or set model_path/dataset_path

app = FastAPI(
    title="T5 Question Generation API",
    description="API for generating technical interview questions using T5 model with RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class JobDescription(BaseModel):
    description: str
    num_questions: int = 3

class QuestionResponse(BaseModel):
    questions: List[str]
    
class Resume(BaseModel):
    # BLOB file PDF binary data
    file: bytes
    JobDescription: str

class ResumeMatchingResponse(BaseModel):
    matchedSkills: List[str]
    MissingSkills: List[str]
    ResumeScore: float


@app.post("/generate-questions", response_model=QuestionResponse)
async def generate_questions(job_description: JobDescription):
    try:
        logger.info(f"Received request for job description: {job_description.description[:100]}...")
        questions = generate_question(
            job_description.description,
            num_questions=job_description.num_questions
        )
        logger.info(f"Successfully generated {len(questions)} questions")
        return QuestionResponse(questions=questions)
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/resume-matching", response_model=ResumeMatchingResponse)
async def resume_matching(resume: Resume):
    try:
        logger.info(f"Received request for resume matching: {resume.file[:100]}...")
        matches = match_resume_to_jd(resume.file, resume.JobDescription)
        logger.info(f"Successfully found {len(matches)} matching jobs")
        return ResumeMatchingResponse(matchedSkills=matches.get("matched_skills", []),
                                       MissingSkills=matches.get("missing_skills", []),
                                       ResumeScore=matches.get("matching_score", 0.0))
    except Exception as e:
        logger.error(f"Error matching resume: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the T5 Question Generation API with RAG"}

if __name__ == "__main__":
    try:
        logger.info("Starting API server...")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}") 