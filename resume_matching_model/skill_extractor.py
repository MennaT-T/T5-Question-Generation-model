import torch
from transformers import DebertaForTokenClassification, DebertaTokenizerFast
from nltk import sent_tokenize, word_tokenize
import nltk


class SkillExtractor:
    def __init__(self, model_path="resume_matching_model/skill_extractor_model"):
        # Ensure NLTK data is downloaded
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        
        # Initialize tokenizer and model
        self.tokenizer = DebertaTokenizerFast.from_pretrained(
            model_path, 
            add_prefix_space=True
        )
        self.model = DebertaForTokenClassification.from_pretrained(model_path)
        self.model.eval()  # Set evaluation mode
        
        # Device configuration (use GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def extract_skills(self, text, confidence_threshold=0.7):
        """Extract skills from text using NER with confidence filtering"""
        sentences = sent_tokenize(text)
        skills = set()
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            inputs = self.tokenizer(
                tokens,
                is_split_into_words=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Process predictions
            probabilities = torch.softmax(outputs.logits, dim=2)[0]
            predictions = outputs.logits.argmax(dim=2)[0]
            word_ids = inputs.word_ids()
            
            current_skill = []
            current_confidences = []
            previous_word_id = None
            
            for idx, pred in enumerate(predictions):
                word_id = word_ids[idx]
                if word_id is None:  # Skip special tokens
                    continue
                
                # Handle subword tokens
                if word_id == previous_word_id:
                    continue
                
                confidence = probabilities[idx][pred].item()
                
                # B-skill tag (start of new skill)
                if pred == 1:
                    self._finalize_skill(current_skill, current_confidences, skills, confidence_threshold)
                    current_skill = [tokens[word_id]]
                    current_confidences = [confidence]
                
                # I-skill tag (continuation of skill)
                elif pred == 2 and current_skill:
                    current_skill.append(tokens[word_id])
                    current_confidences.append(confidence)
                
                # Other tag
                else:
                    self._finalize_skill(current_skill, current_confidences, skills, confidence_threshold)
                    current_skill = []
                    current_confidences = []
                
                previous_word_id = word_id
            
            # Finalize last skill in sentence
            self._finalize_skill(current_skill, current_confidences, skills, confidence_threshold)
        
        return list(skills)
    
    def _finalize_skill(self, tokens, confidences, skills_set, threshold):
        """Helper to finalize and add skill if valid"""
        if not tokens or not confidences:
            return
        
        avg_confidence = sum(confidences) / len(confidences)
        if avg_confidence >= threshold:
            skills_set.add(" ".join(tokens))