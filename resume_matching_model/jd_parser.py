import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import fuzz
import pickle
import logging
from collections import defaultdict


class JobDescriptionParser:
    """
    Robust Job Description parser that extracts sections and their content
    from unstructured JD text, with ML-enhanced detection capabilities.
    """
    
    def __init__(self, fuzzy_threshold: int = 75, model_path: Optional[str] = None):
        self.fuzzy_threshold = fuzzy_threshold
        self.logger = logging.getLogger(__name__)
        
        # JD-specific section patterns
        self.section_patterns = {
            'responsibilities': [
                r'\b(responsibilities|duties|role|tasks|what you.?ll do)\b',
                r'\b(job responsibilities|key responsibilities|main duties)\b',
                r'\b(your role|day to day|daily tasks|primary duties)\b',
                r'\b(you will|you.?ll be responsible|your responsibilities)\b'
            ],
            'requirements': [
                r'\b(requirements|qualifications|skills|must have|prerequisites)\b',
                r'\b(required skills|required qualifications|minimum requirements)\b',
                r'\b(what we.?re looking for|we are looking for|ideal candidate)\b',
                r'\b(essential skills|mandatory|must haves)\b'
            ],
            'preferred_qualifications': [
                r'\b(preferred|nice to have|bonus|plus|advantage|desirable)\b',
                r'\b(preferred qualifications|preferred skills|would be nice)\b',
                r'\b(additional skills|extra points|preferred experience)\b',
                r'\b(nice to haves|good to have|beneficial)\b'
            ],
            'benefits': [
                r'\b(benefits|perks|compensation|package|offer)\b',
                r'\b(what we offer|employee benefits|compensation package)\b',
                r'\b(perks and benefits|benefits package|why join us)\b'
            ],
            'about_company': [
                r'\b(about us|about the company|company overview|who we are)\b',
                r'\b(our company|the company|organization|about)\b',
                r'\b(company profile|company background|our mission)\b'
            ],
            'job_summary': [
                r'\b(job summary|overview|description|position summary)\b',
                r'\b(role overview|position overview|job overview)\b',
                r'\b(summary|introduction|about the role)\b'
            ],
            'experience': [
                r'\b(experience|years|background|expertise)\b',
                r'\b(work experience|professional experience|relevant experience)\b',
                r'\b(minimum experience|required experience)\b'
            ],
            'location': [
                r'\b(location|based|office|remote|hybrid)\b',
                r'\b(work location|job location|position location)\b'
            ],
            'employment_type': [
                r'\b(employment type|job type|position type|contract)\b',
                r'\b(full time|part time|permanent|temporary|freelance)\b'
            ]
        }
        
        # Section variants for fuzzy matching
        self.section_variants = {
            'responsibilities': [
                'responsibilities', 'duties', 'role', 'tasks', 'what you will do',
                'job responsibilities', 'key responsibilities', 'main duties',
                'your role', 'day to day', 'primary responsibilities'
            ],
            'requirements': [
                'requirements', 'qualifications', 'skills required', 'must have',
                'required skills', 'minimum requirements', 'essential skills',
                'what we are looking for', 'ideal candidate', 'prerequisites'
            ],
            'preferred_qualifications': [
                'preferred', 'nice to have', 'bonus', 'desirable', 'advantage',
                'preferred qualifications', 'preferred skills', 'nice to haves',
                'additional skills', 'good to have', 'would be nice'
            ],
            'benefits': [
                'benefits', 'perks', 'compensation', 'package', 'what we offer',
                'employee benefits', 'compensation package', 'perks and benefits'
            ],
            'about_company': [
                'about us', 'about the company', 'company overview', 'who we are',
                'our company', 'company profile', 'about', 'our mission'
            ],
            'job_summary': [
                'job summary', 'overview', 'description', 'position summary',
                'role overview', 'summary', 'about the role', 'introduction'
            ],
            'experience': [
                'experience', 'background', 'work experience', 'years',
                'professional experience', 'minimum experience', 'required experience'
            ],
            'location': [
                'location', 'based', 'office', 'remote', 'hybrid',
                'work location', 'job location', 'position location'
            ],
            'employment_type': [
                'employment type', 'job type', 'full time', 'part time',
                'contract', 'permanent', 'temporary', 'freelance'
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
        Generate training data from JD JSON dataset.
        Creates synthetic section headers and content for training.
        """
        training_data = []
        
        for jd in json_data:
            # Generate section headers for each section type
            
            # Requirements section
            if 'requirements' in jd:
                requirements_variants = [
                    'REQUIREMENTS',
                    'Required Skills',
                    'Qualifications',
                    'MUST HAVE SKILLS',
                    'Essential Requirements',
                    'What We\'re Looking For',
                    'Required Qualifications',
                    'Minimum Requirements'
                ]
                for variant in requirements_variants:
                    training_data.append({
                        'text': variant,
                        'label': 'requirements',
                        'is_header': True,
                        'features': self._extract_line_features(variant)
                    })
            
            # Responsibilities section
            if 'responsibilities' in jd:
                responsibilities_variants = [
                    'RESPONSIBILITIES',
                    'Key Responsibilities',
                    'Your Role',
                    'DUTIES',
                    'What You\'ll Do',
                    'Primary Responsibilities',
                    'Job Responsibilities',
                    'Main Duties'
                ]
                for variant in responsibilities_variants:
                    training_data.append({
                        'text': variant,
                        'label': 'responsibilities',
                        'is_header': True,
                        'features': self._extract_line_features(variant)
                    })
            
            # Experience section (derived from experience_level)
            if 'experience_level' in jd:
                experience_variants = [
                    'EXPERIENCE',
                    'Required Experience',
                    'Work Experience',
                    'BACKGROUND',
                    'Professional Experience',
                    'Minimum Experience',
                    'Years of Experience'
                ]
                for variant in experience_variants:
                    training_data.append({
                        'text': variant,
                        'label': 'experience',
                        'is_header': True,
                        'features': self._extract_line_features(variant)
                    })
            
            # Benefits section (generic since not in dataset)
            benefits_variants = [
                'BENEFITS',
                'What We Offer',
                'Compensation Package',
                'PERKS',
                'Employee Benefits',
                'Package & Benefits'
            ]
            for variant in benefits_variants:
                training_data.append({
                    'text': variant,
                    'label': 'benefits',
                    'is_header': True,
                    'features': self._extract_line_features(variant)
                })
            
            # About Company section
            company_variants = [
                'ABOUT US',
                'Company Overview',
                'About the Company',
                'WHO WE ARE',
                'Our Company',
                'Company Profile'
            ]
            for variant in company_variants:
                training_data.append({
                    'text': variant,
                    'label': 'about_company',
                    'is_header': True,
                    'features': self._extract_line_features(variant)
                })
            
            # Generate negative examples (content, not headers)
            content_examples = [
                f"Position: {jd['job_title']}",
                f"Experience Level: {jd['experience_level']}",
                f"Must have experience with {', '.join(jd['requirements']['must_have'][:3])}",
                f"Nice to have: {', '.join(jd['requirements']['nice_to_have'][:3])}",
                jd['responsibilities'][0] if jd['responsibilities'] else "Handle various responsibilities",
                f"Key skills: {', '.join(jd['target_skills'][:4])}"
            ]
            
            for content in content_examples:
                if content:  # Skip empty content
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
        
        # JD-specific features
        features['likely_short_header'] = len(line.split()) <= 5
        features['has_bullet'] = line.strip().startswith(('•', '●', '-', '*'))
        features['has_jd_keywords'] = any(word.lower() in line.lower() 
                                         for word in ['requirements', 'responsibilities', 'skills', 
                                                     'experience', 'qualifications', 'benefits'])
        
        # Pattern-based features for JD sections
        features['matches_requirements_pattern'] = bool(re.search(
            r'\b(requirements|qualifications|skills|must have)\b', line, re.IGNORECASE))
        features['matches_responsibilities_pattern'] = bool(re.search(
            r'\b(responsibilities|duties|role|tasks)\b', line, re.IGNORECASE))
        features['matches_benefits_pattern'] = bool(re.search(
            r'\b(benefits|perks|compensation|offer)\b', line, re.IGNORECASE))
        features['matches_company_pattern'] = bool(re.search(
            r'\b(about us|company|who we are)\b', line, re.IGNORECASE))
        features['matches_experience_pattern'] = bool(re.search(
            r'\b(experience|years|background)\b', line, re.IGNORECASE))
        
        # Linguistic features
        features['avg_word_length'] = np.mean([len(word) for word in line.split()]) if line.split() else 0
        features['punctuation_ratio'] = len(re.findall(r'[^\w\s]', line)) / len(line) if line else 0
        
        return features
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training."""
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
                int(feature_dict['has_bullet']),
                int(feature_dict['matches_requirements_pattern']),
                int(feature_dict['matches_responsibilities_pattern']),
                int(feature_dict['matches_benefits_pattern']),
                int(feature_dict['matches_company_pattern']),
                int(feature_dict['matches_experience_pattern']),
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
        """Train the ML model on JD dataset."""
        self.logger.info("Generating training data from JD JSON dataset...")
        
        # Generate training data
        df = self.generate_training_data_from_json(json_data)
        self.logger.info(f"Generated {len(df)} training samples")
        
        # Prepare features and labels
        X, y = self.prepare_training_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        self.logger.info(f"Model trained with accuracy: {accuracy:.4f}")
        
        return {'accuracy': accuracy}
    
    def detect_section_header(self, line: str) -> Optional[str]:
        """Detect if a line is a section header and return the section type."""
        line_clean = line.strip()
        if not line_clean or len(line_clean) > 80:  # Too long to be a header
            return None
        
        # Check formatting indicators
        formatting_score = self._calculate_formatting_score(line_clean)
        
        # If we have trained ML model, use it
        if self.is_trained:
            ml_section, ml_confidence = self._predict_section_ml(line_clean)
            if ml_section != 'content' and ml_confidence > 0.6 and formatting_score >= 1:
                return ml_section
        
        # First try exact pattern matching
        detected_section = self._pattern_match_section(line_clean)
        if detected_section and formatting_score >= 2:
            return detected_section
        
        # Then try fuzzy matching
        fuzzy_section = self._fuzzy_match_section(line_clean)
        if fuzzy_section and formatting_score >= 1:
            return fuzzy_section
        
        return None
    
    def _predict_section_ml(self, line: str) -> Tuple[str, float]:
        """Predict section type using ML model."""
        if not self.is_trained:
            return 'content', 0.0
        
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
            int(features['has_bullet']),
            int(features['matches_requirements_pattern']),
            int(features['matches_responsibilities_pattern']),
            int(features['matches_benefits_pattern']),
            int(features['matches_company_pattern']),
            int(features['matches_experience_pattern']),
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
    
    def _calculate_formatting_score(self, line: str) -> int:
        """Calculate formatting score based on common JD header patterns."""
        score = 0
        
        # Common header formatting patterns
        if line.isupper():  # ALL CAPS
            score += 2
        elif line.istitle():  # Title Case
            score += 1
        
        if line.endswith(':'):  # Colon endings
            score += 2
        
        if len(line.split()) <= 5:  # Short headers (1-5 words)
            score += 1
        
        # Markdown or formatting indicators
        if any(line.startswith(marker) for marker in ['**', '##', '__', '*', '###']):
            score += 2
        
        # Bold or underline patterns
        if re.search(r'\*\*.*\*\*|__.*__|<b>.*</b>|<strong>.*</strong>', line):
            score += 2
        
        # Check if line has minimal punctuation (except colon)
        punct_count = len(re.findall(r'[^\w\s:]', line))
        if punct_count == 0 or (punct_count == 1 and line.endswith(':')):
            score += 1
        
        # JD specific formatting
        if re.search(r'^[A-Z\s]{3,}$', line):  # All caps with spaces
            score += 1
        
        return score
    
    def _pattern_match_section(self, line: str) -> Optional[str]:
        """Match section using regex patterns."""
        for section, patterns in self.section_patterns.items():
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    return section
        return None
    
    def _fuzzy_match_section(self, line: str) -> Optional[str]:
        """Match section using fuzzy string matching."""
        line_lower = line.lower().strip(':').strip()
        
        best_match = None
        best_score = 0
        
        for section, variants in self.section_variants.items():
            for variant in variants:
                score = fuzz.ratio(line_lower, variant.lower())
                if score > best_score and score >= self.fuzzy_threshold:
                    best_score = score
                    best_match = section
        
        return best_match
    
    def _enhance_detection_with_context(self, lines: List[str]) -> List[Tuple[int, str]]:
        """Enhance header detection using document context."""
        detected_headers = []
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean:
                continue
            
            # Get context
            prev_line = lines[i-1].strip() if i > 0 else ""
            next_line = lines[i+1].strip() if i < len(lines)-1 else ""
            
            # Calculate context score
            context_score = 0
            
            # Empty line before (section break)
            if not prev_line:
                context_score += 1
            
            # Next line contains content patterns or bullets
            if next_line and (self._detect_jd_content_pattern(next_line) or next_line.startswith(('•', '-', '*'))):
                context_score += 2
            
            # Line is significantly shorter than surrounding text
            avg_surrounding_length = (len(prev_line) + len(next_line)) / 2
            if avg_surrounding_length > 0 and len(line_clean) < avg_surrounding_length * 0.6:
                context_score += 1
            
            # Detect section header
            section = self.detect_section_header(line_clean)
            if section and context_score >= 1:
                detected_headers.append((i, section))
        
        return detected_headers
    
    def _detect_jd_content_pattern(self, line: str) -> bool:
        """Detect if line contains typical JD content."""
        content_indicators = [
            r'\b\d+\+?\s*(years?|yrs?)\b',  # "3+ years", "5 years"
            r'\b(bachelor|master|phd|degree|diploma)\b',  # Education levels
            r'\b(experience|knowledge|proficiency|familiar)\b',  # Experience terms
            r'\b(responsible|manage|develop|implement|design|lead)\b',  # Action verbs
            r'\b(skills?|technologies|tools|frameworks)\b',  # Skill terms
            r'\b(remote|onsite|hybrid|full-time|part-time)\b',  # Work arrangements
            r'\b(competitive|salary|benefits|health|401k)\b',  # Compensation terms
        ]
        
        return any(re.search(pattern, line, re.IGNORECASE) for pattern in content_indicators)
    
    def extract_sections(self, jd_text: str) -> Dict[str, str]:
        """Extract sections from job description text."""
        if not jd_text.strip():
            return {}
        
        lines = jd_text.split('\n')
        
        # Enhance detection with context
        detected_headers = self._enhance_detection_with_context(lines)
        
        if not detected_headers:
            self.logger.warning("No section headers detected")
            return {'content': jd_text}
        
        # Extract sections
        sections = {}
        
        for i, (line_idx, section_name) in enumerate(detected_headers):
            # Determine content boundaries
            start_idx = line_idx + 1
            
            # Find end of section (next header or end of document)
            if i < len(detected_headers) - 1:
                end_idx = detected_headers[i + 1][0]
            else:
                end_idx = len(lines)
            
            # Extract content
            content_lines = []
            for j in range(start_idx, end_idx):
                line = lines[j].strip()
                if line:  # Skip empty lines
                    content_lines.append(line)
            
            # Store section content
            if content_lines:
                # If section already exists, append content
                if section_name in sections:
                    sections[section_name] += '\n' + '\n'.join(content_lines)
                else:
                    sections[section_name] = '\n'.join(content_lines)
        
        # Handle any content before first section
        if detected_headers and detected_headers[0][0] > 0:
            header_content = []
            for i in range(detected_headers[0][0]):
                line = lines[i].strip()
                if line:
                    header_content.append(line)
            if header_content:
                sections['header'] = '\n'.join(header_content)
        
        return sections
    
    def parse_job_description(self, jd_text: str) -> Dict[str, str]:
        """
        Main parsing method that extracts all sections from job description.
        
        Args:
            jd_text: Raw job description text
            
        Returns:
            Dictionary with section names as keys and content as values
        """
        try:
            sections = self.extract_sections(jd_text)
            
            # Log parsing results
            self.logger.info(f"Successfully parsed {len(sections)} sections: {list(sections.keys())}")
            
            return sections
        
        except Exception as e:
            self.logger.error(f"Error parsing job description: {str(e)}")
            return {'error': f"Parsing failed: {str(e)}"}
    
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