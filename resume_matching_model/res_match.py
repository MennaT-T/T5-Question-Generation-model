import base64
from transformers import DebertaForTokenClassification, DebertaTokenizerFast
import fitz  # PyMuPDF
import nltk
from unidecode import unidecode
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
import io

nltk.download('punkt')
nltk.download('punkt_tab')

class SkillMatcher:
    def __init__(self, model_path="resume_matching_model/skill_extractor_model"):
        self.tokenizer = DebertaTokenizerFast.from_pretrained(model_path, add_prefix_space=True)
        self.model = DebertaForTokenClassification.from_pretrained(model_path)

    def _extract_text_from_pdf_binary(self, binary_pdf):
        file_stream = io.BytesIO(binary_pdf)
        doc = fitz.open(stream=file_stream, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return unidecode(text)

    def _extract_skills(self, text):
        sentences = sent_tokenize(text)
        skills = set()
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            inputs = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(dim=2)[0]
            word_ids = inputs.word_ids()
            current_skill = []
            for idx, pred in enumerate(predictions):
                if word_ids[idx] is None:
                    continue
                if pred == 1:  # B-skill
                    if current_skill:
                        skills.add(" ".join(current_skill))
                    current_skill = [tokens[word_ids[idx]]]
                elif pred == 2 and current_skill:  # I-skill
                    current_skill.append(tokens[word_ids[idx]])
                else:
                    if current_skill:
                        skills.add(" ".join(current_skill))
                        current_skill = []
            if current_skill:
                skills.add(" ".join(current_skill))
        return skills

    def _coverage_similarity(self, resume_skills, jd_skills):
        if not jd_skills:
            return 0.0
        matched = jd_skills.intersection(resume_skills)
        return len(matched) / len(jd_skills)

    def match_resume_to_jd(self, resume_pdf_binary, jd_text):
        resume_text = self._extract_text_from_pdf_binary(resume_pdf_binary)
        resume_skills = self._extract_skills(resume_text)
        jd_skills = self._extract_skills(jd_text)

        matched_skills = resume_skills.intersection(jd_skills)
        missing_skills = jd_skills.difference(resume_skills)
        similarity_score = self._coverage_similarity(resume_skills, jd_skills)

        return {
            "matched_skills": list(matched_skills),
            "missing_skills": list(missing_skills),
            "matching_score": round(similarity_score, 2)
        }
        
def match_resume_to_jd(resume_pdf_bytes, jd_text):
    matcher = SkillMatcher()
    try:
        resume_pdf_binary = base64.b64decode(resume_pdf_bytes) 
    except Exception as e:
        raise ValueError("Invalid PDF binary data") from e
    return matcher.match_resume_to_jd(resume_pdf_binary, jd_text)
