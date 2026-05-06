def _get_assessment_sections(self):
        """Gets a SectionList of all Sections currently known to this AssessmentTaken"""
        section_list = []
        for section_idstr in self._my_map['sections']:
            section_list.append(self._get_assessment_section(Id(section_idstr)))
        return AssessmentSectionList(section_list, runtime=self._runtime, proxy=self._proxy)