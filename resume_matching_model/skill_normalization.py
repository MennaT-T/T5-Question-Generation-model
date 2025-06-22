import re
from typing import Dict, List, Set
from difflib import SequenceMatcher

class SkillNormalizer:
    def __init__(self):
        # Define canonical skill mappings
        self.skill_mappings = {
            # Programming Languages
            'python': ['python', 'python language', 'python programming', 'python3', 'py'],
            'javascript': ['javascript', 'js', 'javascript language', 'javascript programming', 'ecmascript'],
            'java': ['java', 'java language', 'java programming', 'oracle java'],
            'c++': ['c++', 'cpp', 'c plus plus', 'cplusplus'],
            'c#': ['c#', 'csharp', 'c sharp', 'c-sharp'],
            'typescript': ['typescript', 'ts', 'typescript language'],
            'php': ['php', 'php language', 'php programming'],
            'ruby': ['ruby', 'ruby language', 'ruby programming'],
            'go': ['go', 'golang', 'go language'],
            'rust': ['rust', 'rust language', 'rust programming'],
            'kotlin': ['kotlin', 'kotlin language'],
            'swift': ['swift', 'swift language', 'swift programming'],
            'scala': ['scala', 'scala language'],
            'r': ['r', 'r language', 'r programming', 'r statistical'],
            
            # Databases
            'postgresql': ['postgresql', 'postgres', 'postgre', 'psql', 'postgres sql'],
            'mysql': ['mysql', 'my sql', 'mysql database'],
            'mongodb': ['mongodb', 'mongo', 'mongo db'],
            'sqlite': ['sqlite', 'sql lite', 'sqlite3'],
            'oracle': ['oracle', 'oracle database', 'oracle db'],
            'sql server': ['sql server', 'microsoft sql server', 'mssql', 'ms sql'],
            'redis': ['redis', 'redis database'],
            'cassandra': ['cassandra', 'apache cassandra'],
            'elasticsearch': ['elasticsearch', 'elastic search', 'es'],
            
            # Frameworks & Libraries
            'react': ['react', 'reactjs', 'react.js', 'react js'],
            'angular': ['angular', 'angularjs', 'angular js', 'angular.js'],
            'vue': ['vue', 'vuejs', 'vue.js', 'vue js'],
            'node.js': ['node.js', 'nodejs', 'node js', 'node'],
            'express': ['express', 'expressjs', 'express.js', 'express js'],
            'django': ['django', 'django framework'],
            'flask': ['flask', 'flask framework'],
            'spring': ['spring', 'spring framework', 'spring boot'],
            'laravel': ['laravel', 'laravel framework'],
            'rails': ['rails', 'ruby on rails', 'ror'],
            'jquery': ['jquery', 'j query', 'jquery library'],
            'bootstrap': ['bootstrap', 'bootstrap css', 'twitter bootstrap'],
            'tensorflow': ['tensorflow', 'tensor flow', 'tf'],
            'pytorch': ['pytorch', 'py torch', 'torch'],
            'scikit-learn': ['scikit-learn', 'sklearn', 'scikit learn'],
            'pandas': ['pandas', 'pandas library'],
            'numpy': ['numpy', 'num py'],
            
            # Cloud & DevOps
            'aws': ['aws', 'amazon web services', 'amazon aws'],
            'azure': ['azure', 'microsoft azure', 'ms azure'],
            'gcp': ['gcp', 'google cloud platform', 'google cloud', 'gc platform'],
            'docker': ['docker', 'docker container', 'containerization'],
            'kubernetes': ['kubernetes', 'k8s', 'k8', 'kube'],
            'jenkins': ['jenkins', 'jenkins ci', 'jenkins ci/cd'],
            'git': ['git', 'git version control', 'git scm'],
            'github': ['github', 'git hub'],
            'gitlab': ['gitlab', 'git lab'],
            'terraform': ['terraform', 'terraform iac'],
            'ansible': ['ansible', 'ansible automation'],
            
            # Operating Systems
            'linux': ['linux', 'gnu/linux', 'unix', 'ubuntu', 'centos', 'rhel'],
            'windows': ['windows', 'microsoft windows', 'ms windows'],
            'macos': ['macos', 'mac os', 'osx', 'os x', 'mac osx'],
            
            # Web Technologies
            'html': ['html', 'html5', 'hypertext markup language'],
            'css': ['css', 'css3', 'cascading style sheets'],
            'rest api': ['rest api', 'restful api', 'rest', 'restful'],
            'graphql': ['graphql', 'graph ql'],
            'json': ['json', 'javascript object notation'],
            'xml': ['xml', 'extensible markup language'],
            
            # Testing
            'junit': ['junit', 'j unit'],
            'selenium': ['selenium', 'selenium webdriver'],
            'jest': ['jest', 'jest testing'],
            'pytest': ['pytest', 'py test'],
            
            # Other Tools
            'photoshop': ['photoshop', 'adobe photoshop', 'ps'],
            'illustrator': ['illustrator', 'adobe illustrator', 'ai'],
            'figma': ['figma', 'figma design'],
            'sketch': ['sketch', 'sketch app'],
            'jira': ['jira', 'atlassian jira'],
            'confluence': ['confluence', 'atlassian confluence'],
            'slack': ['slack', 'slack communication'],
            'microsoft office': ['microsoft office', 'ms office', 'office suite', 'excel', 'word', 'powerpoint'],
        }
        
        # Create reverse mapping for quick lookup
        self.reverse_mapping = {}
        for canonical, variants in self.skill_mappings.items():
            for variant in variants:
                self.reverse_mapping[variant.lower()] = canonical
        
        # Common words to remove during preprocessing
        self.stop_words = {'language', 'programming', 'framework', 'library', 'database', 'tool', 'software'}
    
    def preprocess_skill(self, skill: str) -> str:
        """Preprocess skill name for better matching"""
        if not skill:
            return ""
        
        # Convert to lowercase and strip whitespace
        skill = skill.lower().strip()
        
        # Remove special characters except dots, hashes, and plus signs
        skill = re.sub(r'[^\w\s.#+\-]', ' ', skill)
        
        # Normalize whitespace
        skill = re.sub(r'\s+', ' ', skill).strip()
        
        return skill
    
    def normalize_skill(self, skill: str) -> str:
        """
        Normalize a skill name to its canonical form
        """
        if not skill:
            return ""
        
        # Preprocess the skill
        processed_skill = self.preprocess_skill(skill)
        
        # Direct lookup in reverse mapping
        if processed_skill in self.reverse_mapping:
            return self.reverse_mapping[processed_skill]
        
        # Try without common stop words
        words = processed_skill.split()
        filtered_words = [w for w in words if w not in self.stop_words]
        if filtered_words:
            filtered_skill = ' '.join(filtered_words)
            if filtered_skill in self.reverse_mapping:
                return self.reverse_mapping[filtered_skill]
        
        # Try fuzzy matching with similarity threshold
        best_match = self.find_best_fuzzy_match(processed_skill)
        if best_match:
            return best_match
        
        # If no match found, return the preprocessed skill as canonical
        return processed_skill
    
    def find_best_fuzzy_match(self, skill: str, threshold: float = 0.8) -> str:
        """
        Find the best fuzzy match for a skill using string similarity
        """
        best_score = 0
        best_match = None
        
        for variant, canonical in self.reverse_mapping.items():
            # Calculate similarity ratio
            similarity = SequenceMatcher(None, skill, variant).ratio()
            
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = canonical
        
        return best_match
    
    def normalize_skills_list(self, skills: List[str]) -> List[str]:
        """
        Normalize a list of skills and remove duplicates
        """
        normalized = []
        seen = set()
        
        for skill in skills:
            normalized_skill = self.normalize_skill(skill)
            if normalized_skill and normalized_skill not in seen:
                normalized.append(normalized_skill)
                seen.add(normalized_skill)
        
        return normalized
    
    def add_skill_mapping(self, canonical: str, variants: List[str]):
        """
        Add a new skill mapping or extend existing one
        """
        canonical = canonical.lower()
        variants = [v.lower() for v in variants]
        
        if canonical in self.skill_mappings:
            # Extend existing mapping
            existing_variants = set(self.skill_mappings[canonical])
            existing_variants.update(variants)
            self.skill_mappings[canonical] = list(existing_variants)
        else:
            # Add new mapping
            self.skill_mappings[canonical] = variants
        
        # Update reverse mapping
        for variant in variants:
            self.reverse_mapping[variant] = canonical
    
    def skill_similarity(self, res_skills: List[str], jd_skills: List[str]) -> dict:
        """
        Calculate similarity between two skill lists after normalization
        """
        normalized_res_skills = set(self.normalize_skills_list(res_skills))
        normalized_jd_skills = set(self.normalize_skills_list(jd_skills))
        
        if not normalized_res_skills and not normalized_jd_skills:
            return {"matching_score": 1.0}

        if not normalized_res_skills or not normalized_jd_skills:
            return {"matching_score": 0.0}

        matched_skills = normalized_res_skills.intersection(normalized_jd_skills)
        missing_skills = normalized_jd_skills.difference(normalized_res_skills)
        
        # union = len(normalized_skills1.union(normalized_skills2))
        # return intersection / union if union > 0 else 0.0
        
        # return coverage
        # return round(intersection / len(normalized_jd_skills), 2)
        similarity_score = round(len(matched_skills) / len(normalized_jd_skills), 2)
        return  {
            "matched_skills": list(matched_skills), # intersection
            "missing_skills": list(missing_skills), # difference
            "matching_score": similarity_score # converage
        }
