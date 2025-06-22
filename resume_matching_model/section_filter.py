from typing import Dict, List

class SectionFilter:
    def __init__(
        self,
        resume_targets: List[str] = None,
        jd_targets: List[str] = None
    ):
        self.resume_targets = resume_targets or ["skills", "projects", "experience"]
        self.jd_targets = jd_targets or ["requirements", "responsibilities"]

    def filter_resume(self, resume_data: Dict[str, str], targets: List[str] = None) -> str:
        return self._merge_sections(resume_data, targets or self.resume_targets)

    def filter_jd(self, jd_data: Dict[str, str], targets: List[str] = None) -> str:
        return self._merge_sections(jd_data, targets or self.jd_targets)

    def _merge_sections(self, data: Dict[str, str], section_keys: List[str]) -> str:
        merged = []
        for key in section_keys:
            value = data.get(key)
            if value:
                merged.append(f"{key}:\n{value}")
        return "\n".join(merged)