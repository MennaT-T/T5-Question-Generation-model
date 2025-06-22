from .ml_enhanced_parser import MLEnhancedResumeParser
from .jd_parser import JobDescriptionParser
from .section_filter import SectionFilter
from .skill_extractor import SkillExtractor
from .skill_normalization import SkillNormalizer
import io
import base64
from unidecode import unidecode
import fitz  # PyMuPDF

class PipelineProcessor:
    def __init__(self):
        # Initialize the components of the pipeline
        self.resume_parser = MLEnhancedResumeParser()
        self.jd_parser = JobDescriptionParser()
        self.section_filter = SectionFilter()
        self.skill_extractor = SkillExtractor()
        self.skill_normalizer = SkillNormalizer()

    def _extract_text_from_pdf_binary(self, resume_base64_string):
        """Extract text from base64 encoded PDF binary data."""
        pdf_data = base64.b64decode(resume_base64_string)
        file_stream = io.BytesIO(pdf_data)
        doc = fitz.open(stream=file_stream, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return unidecode(text)

    def process(self, resume_input: str, jd_text: str) -> dict:
        """
        Execute the pipeline to process resumes and job descriptions.

        :param resume_text: Raw text of the resume
        :param jd_text: Raw text of the job description
        :return: Dictionary with pipeline results
        """

        # Step 0: Extract text from PDF if needed
        resume_text = self._extract_text_from_pdf_binary(resume_input)

        # Step 1: Parsing
        parsed_resume = self.resume_parser.parse_resume(resume_text)
        parsed_jd = self.jd_parser.parse_job_description(jd_text)

        # Step 2: Section Filtering
        filtered_resume = self.section_filter.filter_resume(parsed_resume)
        filtered_jd = self.section_filter.filter_jd(parsed_jd)

        # Step 3: Skill Extraction
        extracted_resume_skills = self.skill_extractor.extract_skills(filtered_resume)
        extracted_jd_skills = self.skill_extractor.extract_skills(filtered_jd)

        # Step 4: Normalization and Matching
        similarity_results = self.skill_normalizer.skill_similarity(extracted_resume_skills, extracted_jd_skills)

        # Structure of similarity results:
        # {
        #     "matched_skills": list,  # Skills that match (intersection)
        #     "missing_skills": list,  # Skills that are missing (difference)
        #     "matching_score": float  # Matching score based on coverage
        # }

        # Compile results
        return similarity_results


def test():
    """
    Test function to verify the pipeline works as expected.
    """
    resume_input = "JVBERi0xLjcKJcK1wrYKCjEgMCBvYmoKPDwvVHlwZS9DYXRhbG9nL1BhZ2VzIDIgMCBSPj4KZW5kb2JqCgoyIDAgb2JqCjw8L1R5cGUvUGFnZXMvQ291bnQgMS9LaWRzWzQgMCBSXT4+CmVuZG9iagoKMyAwIG9iago8PC9Gb250PDwvaGVsdiA1IDAgUj4+Pj4KZW5kb2JqCgo0IDAgb2JqCjw8L1R5cGUvUGFnZS9NZWRpYUJveFswIDAgNTk1IDg0Ml0vUm90YXRlIDAvUmVzb3VyY2VzIDMgMCBSL1BhcmVudCAyIDAgUi9Db250ZW50c1s2IDAgUl0+PgplbmRvYmoKCjUgMCBvYmoKPDwvVHlwZS9Gb250L1N1YnR5cGUvVHlwZTEvQmFzZUZvbnQvSGVsdmV0aWNhL0VuY29kaW5nL1dpbkFuc2lFbmNvZGluZz4+CmVuZG9iagoKNiAwIG9iago8PC9MZW5ndGggMTExL0ZpbHRlci9GbGF0ZURlY29kZT4+CnN0cmVhbQp42h2MsQrDMAxEd32F/qCWY58SKB0CWboVtJVOjkOHduiS7+8liIN7j0PykznENPFMPat70vjK5d0/u5ppbPq81uSTF4zY0HPLqawwDOQJHZXcUGGeSYfxc1NoGDajMzT+GMDcXnGXJeQhfxAkHMYKZW5kc3RyZWFtCmVuZG9iagoKeHJlZgowIDcKMDAwMDAwMDAwMCA2NTUzNSBmIAowMDAwMDAwMDE2IDAwMDAwIG4gCjAwMDAwMDAwNjIgMDAwMDAgbiAKMDAwMDAwMDExNCAwMDAwMCBuIAowMDAwMDAwMTU1IDAwMDAwIG4gCjAwMDAwMDAyNjIgMDAwMDAgbiAKMDAwMDAwMDM1MSAwMDAwMCBuIAoKdHJhaWxlcgo8PC9TaXplIDcvUm9vdCAxIDAgUi9JRFs8QzNBMDRBQzM4QzJCNjIxQzRCQzNBODZDN0U0QTE1QzM+PDc1OEIwQTJGNjQ2MzhCMzc2Q0M3REMyRUI1MTMyRDE4Pl0+PgpzdGFydHhyZWYKNTMxCiUlRU9GCg=="  # Replace with actual base64 encoded PDF string
    jd_text = "Sample job description text here."

    processor = PipelineProcessor()
    results = processor.process(resume_input, jd_text)

    print("Matched Skills:", results.get("matched_skills", []))
    print("Missing Skills:", results.get("missing_skills", []))
    print("Matching Score:", results.get("matching_score", 0.0))
    
# if __name__ == "__main__":
#     test()  # Run the test function to verify the pipeline