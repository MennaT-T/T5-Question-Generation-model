import json
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import logging
from fuzzywuzzy import fuzz
import joblib

class MLEnhancedResumeParser:
    """
    Enhanced resume parser with machine learning capabilities
    for better section detection and content extraction.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Traditional rule-based patterns (fallback)
        self.section_patterns = {
            'education': [
                r'\b(education|academic|qualification|degree|university|college|school)\b',
                r'\b(educational background|academic background|academic history)\b'
            ],
            'experience': [
                r'\b(experience|employment|work|career|professional)\b',
                r'\b(work experience|professional experience|employment history)\b'
            ],
            'skills': [
                r'\b(skills|competenc|abilities|expertise|proficienc)\b',
                r'\b(technical skills|core competencies|key skills)\b'
            ],
            'projects': [
                r'\b(projects|portfolio|work samples|achievements)\b',
                r'\b(key projects|notable projects|project experience)\b'
            ]
        }
        
        # ML Components
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = None
        self.is_trained = False
        
        # Load pre-trained model if provided
        if model_path:
            self.load_model(model_path)
    
    def generate_training_data_from_json(self, json_data: List[Dict]) -> pd.DataFrame:
        """
        Generate training data from your JSON resume dataset.
        Creates synthetic section headers and content for training.
        """
        training_data = []
        
        for resume in json_data:
            # Generate section headers and labels
            
            # Education section
            if 'education' in resume:
                education_variants = [
                    'EDUCATION',
                    'Educational Background',
                    'Academic Qualifications',
                    'ACADEMIC BACKGROUND',
                    'Education:',
                    'Educational History'
                ]
                for variant in education_variants:
                    training_data.append({
                        'text': variant,
                        'label': 'education',
                        'is_header': True,
                        'features': self._extract_line_features(variant)
                    })
            
            # Work Experience section
            if 'work_experience' in resume:
                experience_variants = [
                    'WORK EXPERIENCE',
                    'Professional Experience',
                    'Employment History',
                    'EXPERIENCE',
                    'Career History',
                    'Professional Background'
                ]
                for variant in experience_variants:
                    training_data.append({
                        'text': variant,
                        'label': 'experience',
                        'is_header': True,
                        'features': self._extract_line_features(variant)
                    })
            
            # Skills section
            if 'skills' in resume:
                skills_variants = [
                    'SKILLS',
                    'Technical Skills',
                    'Core Competencies',
                    'TECHNICAL SKILLS',
                    'Key Skills',
                    'Competencies'
                ]
                for variant in skills_variants:
                    training_data.append({
                        'text': variant,
                        'label': 'skills',
                        'is_header': True,
                        'features': self._extract_line_features(variant)
                    })
            
            # Projects section (from work experience projects)
            if any('projects' in exp for exp in resume.get('work_experience', [])):
                project_variants = [
                    'PROJECTS',
                    'Key Projects',
                    'Notable Projects',
                    'PROJECT EXPERIENCE',
                    'Personal Projects',
                    'Professional Projects'
                ]
                for variant in project_variants:
                    training_data.append({
                        'text': variant,
                        'label': 'projects',
                        'is_header': True,
                        'features': self._extract_line_features(variant)
                    })
            
            # Generate negative examples (non-headers)
            content_examples = [
                f"{resume['name']}",
                f"B.Sc. in Computer Science from {resume['education']['institution']}",
                f"Graduated in {resume['education']['graduation_year']}",
                f"Working at {resume['work_experience'][0]['company']} as {resume['work_experience'][0]['title']}",
                ', '.join(resume['skills'][:5]) if len(resume['skills']) > 5 else ', '.join(resume['skills'])
            ]
            
            for content in content_examples:
                training_data.append({
                    'text': content,
                    'label': 'content',
                    'is_header': False,
                    'features': self._extract_line_features(content)
                })
        
        return pd.DataFrame(training_data)
    
    def _extract_line_features(self, line: str) -> Dict[str, Any]:
        """Extract comprehensive features from a line of text."""
        features = {}
        
        # Basic text features
        features['length'] = len(line)
        features['word_count'] = len(line.split())
        features['char_count'] = len(line.replace(' ', ''))
        
        # Formatting features
        features['is_upper'] = line.isupper()
        features['is_title'] = line.istitle()
        features['is_lower'] = line.islower()
        features['has_colon'] = ':' in line
        features['ends_with_colon'] = line.endswith(':')
        features['has_numbers'] = bool(re.search(r'\d', line))
        features['has_special_chars'] = bool(re.search(r'[^\w\s]', line))
        
        # Positional features (approximated)
        features['likely_short_header'] = len(line.split()) <= 4
        features['has_common_words'] = any(word.lower() in line.lower() 
                                         for word in ['experience', 'education', 'skills', 'projects'])
        
        # Pattern-based features
        features['matches_education_pattern'] = bool(re.search(
            r'\b(education|academic|qualification|degree)\b', line, re.IGNORECASE))
        features['matches_experience_pattern'] = bool(re.search(
            r'\b(experience|employment|work|career)\b', line, re.IGNORECASE))
        features['matches_skills_pattern'] = bool(re.search(
            r'\b(skills|competenc|abilities|expertise)\b', line, re.IGNORECASE))
        features['matches_projects_pattern'] = bool(re.search(
            r'\b(projects|portfolio|achievements)\b', line, re.IGNORECASE))
        
        # Linguistic features
        features['avg_word_length'] = np.mean([len(word) for word in line.split()]) if line.split() else 0
        features['punctuation_ratio'] = len(re.findall(r'[^\w\s]', line)) / len(line) if line else 0
        
        return features
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training."""
        # Extract text features
        texts = df['text'].tolist()
        
        # Create TF-IDF features
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                lowercase=True,
                stop_words='english'
            )
            text_features = self.vectorizer.fit_transform(texts)
        else:
            text_features = self.vectorizer.transform(texts)
        
        # Extract manual features
        manual_features = []
        for _, row in df.iterrows():
            feature_dict = row['features']
            feature_vector = [
                feature_dict['length'],
                feature_dict['word_count'],
                int(feature_dict['is_upper']),
                int(feature_dict['is_title']),
                int(feature_dict['has_colon']),
                int(feature_dict['ends_with_colon']),
                int(feature_dict['has_numbers']),
                int(feature_dict['likely_short_header']),
                int(feature_dict['matches_education_pattern']),
                int(feature_dict['matches_experience_pattern']),
                int(feature_dict['matches_skills_pattern']),
                int(feature_dict['matches_projects_pattern']),
                feature_dict['avg_word_length'],
                feature_dict['punctuation_ratio']
            ]
            manual_features.append(feature_vector)
        
        manual_features = np.array(manual_features)
        
        # Combine features
        combined_features = np.hstack([text_features.toarray(), manual_features])
        
        # Prepare labels
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            labels = self.label_encoder.fit_transform(df['label'])
        else:
            labels = self.label_encoder.transform(df['label'])
        
        return combined_features, labels
    
    def train_model(self, json_data: List[Dict], test_size: float = 0.2) -> Dict[str, float]:
        """Train the ML model on your resume dataset."""
        self.logger.info("Generating training data from JSON dataset...")
        
        # Generate training data
        df = self.generate_training_data_from_json(json_data)
        self.logger.info(f"Generated {len(df)} training samples")
        
        # Prepare features and labels
        X, y = self.prepare_training_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train ensemble of models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'naive_bayes': MultinomialNB()
        }
        
        best_model = None
        best_score = 0
        results = {}
        
        for name, model in models.items():
            self.logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            
            self.logger.info(f"{name} accuracy: {accuracy:.4f}")
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
        
        # Use best model
        self.classifier = best_model
        self.is_trained = True
        
        self.logger.info(f"Best model selected with accuracy: {best_score:.4f}")
        return results
    
    def predict_section(self, line: str) -> Tuple[str, float]:
        """Predict section type for a line using ML model."""
        if not self.is_trained:
            return self._fallback_prediction(line)
        
        # Prepare features
        features = self._extract_line_features(line)
        
        # Text features
        text_features = self.vectorizer.transform([line])
        
        # Manual features
        manual_features = np.array([[
            features['length'],
            features['word_count'],
            int(features['is_upper']),
            int(features['is_title']),
            int(features['has_colon']),
            int(features['ends_with_colon']),
            int(features['has_numbers']),
            int(features['likely_short_header']),
            int(features['matches_education_pattern']),
            int(features['matches_experience_pattern']),
            int(features['matches_skills_pattern']),
            int(features['matches_projects_pattern']),
            features['avg_word_length'],
            features['punctuation_ratio']
        ]])
        
        # Combine features
        combined_features = np.hstack([text_features.toarray(), manual_features])
        
        # Predict
        prediction = self.classifier.predict(combined_features)[0]
        probabilities = self.classifier.predict_proba(combined_features)[0]
        confidence = np.max(probabilities)
        
        # Convert back to label
        section_type = self.label_encoder.inverse_transform([prediction])[0]
        
        return section_type, confidence
    
    def _fallback_prediction(self, line: str) -> Tuple[str, float]:
        """Fallback to rule-based prediction if ML model not available."""
        line_lower = line.lower().strip()
        
        for section, patterns in self.section_patterns.items():
            for pattern in patterns:
                if re.search(pattern, line_lower):
                    return section, 0.8
        
        return 'content', 0.5
    
    def extract_sections_ml(self, resume_text: str, confidence_threshold: float = 0.6) -> Dict[str, str]:
        """Extract sections using ML-enhanced detection."""
        lines = resume_text.split('\n')
        sections = {}
        current_section = None
        current_content = []
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
            
            # Predict section type
            section_type, confidence = self.predict_section(line_clean)
            
            # Determine if this is a header
            is_header = (section_type != 'content' and 
                        confidence >= confidence_threshold and
                        self._is_likely_header(line_clean))
            
            if is_header:
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                current_section = section_type
                current_content = []
            elif current_section:
                current_content.append(line_clean)
        
        # Don't forget the last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _is_likely_header(self, line: str) -> bool:
        """Additional heuristic to determine if line is likely a header."""
        return (len(line.split()) <= 5 and 
                (line.isupper() or line.istitle() or line.endswith(':')))
    
    def save_model(self, filepath: str):
        """Save the trained model and components."""
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data['vectorizer']
            self.classifier = model_data['classifier']
            self.label_encoder = model_data['label_encoder']
            self.is_trained = model_data['is_trained']
            
            self.logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.is_trained = False
    
    def parse_resume(self, resume_text: str, use_hybrid: bool = True) -> Dict[str, str]:
        """
        Main parsing method with hybrid ML + rule-based approach.
        """
        if use_hybrid and self.is_trained:
            return self._hybrid_extraction(resume_text)
        elif self.is_trained:
            return self.extract_sections_ml(resume_text)
        else:
            return self._extract_sections_rule_based(resume_text)
    
    def _hybrid_extraction(self, resume_text: str) -> Dict[str, str]:
        """
        Hybrid approach: Use ML predictions + rule-based validation + fuzzy matching.
        """
        lines = resume_text.split('\n')
        sections = {}
        current_section = None
        current_content = []
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean:
                continue
            
            # Get ML prediction
            ml_section, ml_confidence = self.predict_section(line_clean)
            
            # Get rule-based prediction
            rule_section = self._get_rule_based_section(line_clean)
            
            # Get fuzzy match prediction
            fuzzy_section = self._fuzzy_match_section(line_clean)
            
            # Context analysis
            context_score = self._analyze_context(lines, i)
            
            # Decision logic: Combine all approaches
            is_header, final_section = self._make_final_decision(
                line_clean, ml_section, ml_confidence, rule_section, 
                fuzzy_section, context_score
            )
            
            if is_header:
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                current_section = final_section
                current_content = []
                self.logger.debug(f"Detected header: '{line_clean}' -> {final_section}")
            elif current_section:
                current_content.append(line_clean)
        
        # Don't forget the last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _make_final_decision(self, line: str, ml_section: str, ml_confidence: float,
                           rule_section: Optional[str], fuzzy_section: Optional[str],
                           context_score: int) -> Tuple[bool, Optional[str]]:
        """
        Make final decision on whether line is header and which section.
        """
        # Voting system
        votes = {}
        
        # ML vote (weighted by confidence)
        if ml_section != 'content' and ml_confidence > 0.5:
            weight = ml_confidence * 2  # Give ML higher weight
            votes[ml_section] = votes.get(ml_section, 0) + weight
        
        # Rule-based vote
        if rule_section:
            votes[rule_section] = votes.get(rule_section, 0) + 1.5
        
        # Fuzzy matching vote
        if fuzzy_section:
            votes[fuzzy_section] = votes.get(fuzzy_section, 0) + 1.0
        
        # Context boost
        if votes and context_score >= 2:
            for section in votes:
                votes[section] += 0.5
        
        # Header likelihood based on formatting
        header_likelihood = self._calculate_header_likelihood(line)
        
        # Make decision
        if votes and header_likelihood > 0.3:
            best_section = max(votes, key=votes.get)
            best_score = votes[best_section]
            
            # Threshold for final decision
            if best_score >= 2.0:
                return True, best_section
        
        return False, None
    
    def _calculate_header_likelihood(self, line: str) -> float:
        """Calculate likelihood that line is a header based on formatting."""
        score = 0.0
        
        if line.isupper():
            score += 0.4
        elif line.istitle():
            score += 0.2
        
        if line.endswith(':'):
            score += 0.3
        
        if len(line.split()) <= 4:
            score += 0.2
        
        if not re.search(r'[.!?]', line):  # No sentence-ending punctuation
            score += 0.1
        
        return min(score, 1.0)
    
    def _get_rule_based_section(self, line: str) -> Optional[str]:
        """Get section using rule-based patterns."""
        for section, patterns in self.section_patterns.items():
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    return section
        return None
    
    def _fuzzy_match_section(self, line: str, threshold: int = 75) -> Optional[str]:
        """Match section using fuzzy string matching."""
        section_variants = {
            'education': ['education', 'academic background', 'qualifications'],
            'experience': ['experience', 'work experience', 'employment'],
            'skills': ['skills', 'competencies', 'abilities'],
            'projects': ['projects', 'portfolio', 'achievements']
        }
        
        line_clean = line.lower().strip(':').strip()
        
        for section, variants in section_variants.items():
            for variant in variants:
                if fuzz.ratio(line_clean, variant) >= threshold:
                    return section
        return None
    
    def _analyze_context(self, lines: List[str], current_idx: int) -> int:
        """Analyze context around current line."""
        score = 0
        
        # Check previous line
        if current_idx > 0:
            prev_line = lines[current_idx - 1].strip()
            if not prev_line:  # Empty line before
                score += 1
        
        # Check next line
        if current_idx < len(lines) - 1:
            next_line = lines[current_idx + 1].strip()
            if next_line and self._looks_like_content(next_line):
                score += 1
        
        # Check if current line is short compared to surrounding
        current_line = lines[current_idx].strip()
        surrounding_lengths = []
        
        for i in range(max(0, current_idx - 2), min(len(lines), current_idx + 3)):
            if i != current_idx and lines[i].strip():
                surrounding_lengths.append(len(lines[i].strip()))
        
        if surrounding_lengths:
            avg_length = sum(surrounding_lengths) / len(surrounding_lengths)
            if len(current_line) < avg_length * 0.6:
                score += 1
        
        return score
    
    def _looks_like_content(self, line: str) -> bool:
        """Check if line looks like content rather than header."""
        content_indicators = [
            len(line) > 50,  # Long line
            '.' in line,  # Contains periods
            ',' in line,  # Contains commas
            any(word in line.lower() for word in ['the', 'and', 'with', 'using', 'for']),
            re.search(r'\d{4}', line) is not None,  # Contains year
            '(' in line and ')' in line,  # Contains parentheses
        ]
        
        return sum(content_indicators) >= 2
    
    def _extract_sections_rule_based(self, resume_text: str) -> Dict[str, str]:
        """Rule-based extraction (fallback method)."""
        lines = resume_text.split('\n')
        sections = {}
        current_section = None
        current_content = []
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
            
            # Check if line is a section header using rules
            section_type = None
            for section, patterns in self.section_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line_clean, re.IGNORECASE):
                        if self._is_likely_header(line_clean):
                            section_type = section
                            break
                if section_type:
                    break
            
            if section_type:
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                current_section = section_type
                current_content = []
            elif current_section:
                current_content.append(line_clean)
        
        # Don't forget the last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections

# Training and usage example
def train_and_use_parser():
    """Example of how to train and use the ML-enhanced parser."""
    
    # Load your JSON data
    with open('resume_dataset.json', 'r') as f:
        resume_data = json.load(f)
    
    # Initialize parser
    parser = MLEnhancedResumeParser()
    
    # Train the model
    print("Training ML model...")
    results = parser.train_model(resume_data)
    print("Training results:", results)
    
    # Save the trained model
    parser.save_model('resume_parser_model.pkl')
    
    # Example usage
    sample_resume = """
    JOHN DOE
    Software Engineer
    
    PROFESSIONAL SUMMARY
    Experienced software engineer with 5 years of experience.
    
    WORK EXPERIENCE
    Senior Software Engineer - Tech Corp (2022-2024)
    • Developed scalable web applications
    
    EDUCATION
    Bachelor of Science in Computer Science
    University of Technology, 2020
    
    TECHNICAL SKILLS
    • Python, JavaScript, Java
    • Django, React, Spring Boot
    """
    
    # Parse using ML
    sections = parser.parse_resume(sample_resume, use_ml=True)
    
    print("\nExtracted sections:")
    for section_name, content in sections.items():
        print(f"\n{section_name.upper()}:")
        print(content)

if __name__ == "__main__":
    train_and_use_parser()
