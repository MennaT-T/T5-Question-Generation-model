import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Optional
import pandas as pd
import re

class T5QuestionGenerator:
    _instance = None
    _model = None
    _tokenizer = None
    _device = None
    _vector_store = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(T5QuestionGenerator, cls).__new__(cls)
            cls._instance._initialize_model()
            cls._instance._initialize_vector_store()
        return cls._instance

    def _initialize_model(self):
        """Initialize the T5 model and tokenizer."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "t5_question_gen_model")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model directory not found at: {model_path}")
            self._tokenizer = T5Tokenizer.from_pretrained(model_path)
            self._model = T5ForConditionalGeneration.from_pretrained(model_path)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
            self._model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize T5 model: {str(e)}")

    def _initialize_vector_store(self):
        """Initialize the FAISS vector store with a knowledge base from Excel."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            kb_path = os.path.join(current_dir, "knowledge_base.xlsx")
            if not os.path.exists(kb_path):
                raise FileNotFoundError(f"Knowledge base not found at: {kb_path}")
            
            df = pd.read_excel(kb_path)
            if "job_description" not in df.columns or "questions" not in df.columns:
                raise ValueError("Excel file must have 'job_description' and 'questions' columns")
            
            documents = []
            for _, row in df.iterrows():
                desc = row["job_description"]
                questions = str(row["questions"]).split(",")  # Handle string or list of questions
                text = f"Job Description: {desc}\nQuestions: {', '.join(questions)}"
                documents.append(Document(page_content=text, metadata={"source": desc}))

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self._vector_store = FAISS.from_documents(documents, embeddings)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vector store: {str(e)}")

    def _is_valid_difference_question(self, question: str) -> bool:
        """Check if a 'What is the difference' or 'Explain the difference' question is valid and coherent."""
        # Check for either pattern (case-insensitive)
        if not (question.lower().startswith("what is the difference between") or 
                question.lower().startswith("explain the difference between")):
            return True  # Not a difference question, so it's valid by default
        
        # Extract the part after the pattern
        pattern = r"(?:what is|explain) the difference between (.+)\?"
        match = re.search(pattern, question, re.IGNORECASE)
        if not match:
            return False  # Malformed question
        
        content = match.group(1).strip()
        # Check for two distinct entities separated by 'and'
        entities = [e.strip() for e in content.split("and") if e.strip()]
        if len(entities) != 2 or entities[0].lower() == entities[1].lower():
            return False  # Either not two entities or they're the same
        
        return True

    def _remove_repetitive_questions(self, questions: List[str]) -> List[str]:
        """Remove repetitive questions while preserving order."""
        seen = set()
        unique_questions = []
        for q in questions:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                unique_questions.append(q)
        return unique_questions

    def generate_question(self, job_description: str, num_questions: int = 3) -> List[str]:
        """
        Generate technical interview questions using RAG with improved filtering.
        """
        if not job_description.strip():
            raise ValueError("Job description cannot be empty")
        if num_questions < 1:
            raise ValueError("Number of questions must be at least 1")

        try:
            retriever = self._vector_store.as_retriever(search_kwargs={"k": 2})
            relevant_docs = retriever.get_relevant_documents(job_description)
            context = "\n".join([doc.page_content for doc in relevant_docs])

            input_text = f"Given the following context and job description, generate {num_questions} specific technical interview questions:\nContext: {context}\nJob Description: {job_description}"
            
            inputs = self._tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self._device)

            generated_questions = []
            attempts = 0
            max_attempts = num_questions * 3  # Prevent infinite loops

            while len(generated_questions) < num_questions and attempts < max_attempts:
                with torch.no_grad():
                    outputs = self._model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=128,
                        do_sample=True,
                        top_k=75,
                        top_p=0.90,
                        repetition_penalty=1.2,
                        num_return_sequences=num_questions,
                        early_stopping=True
                    )

                # Decode and process generated questions
                new_questions = [self._tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

                # Filter invalid "What's the difference" or "Explain the difference" questions
                valid_questions = [q for q in new_questions if self._is_valid_difference_question(q)]

                # Add valid questions to the list
                generated_questions.extend(valid_questions)

                # Remove duplicates while preserving order
                generated_questions = self._remove_repetitive_questions(generated_questions)

                attempts += 1

            # If we still don't have enough questions, raise a warning
            if len(generated_questions) < num_questions:
                print(f"Warning: Only generated {len(generated_questions)} valid questions out of {num_questions} requested.")

            # Return the requested number of questions (or all we have)
            return generated_questions[:num_questions]

        except Exception as e:
            raise RuntimeError(f"Error generating questions: {str(e)}")

# Create a singleton instance
question_generator = T5QuestionGenerator()

def generate_question(job_description: str, num_questions: int = 3) -> List[str]:
    return question_generator.generate_question(job_description, num_questions)