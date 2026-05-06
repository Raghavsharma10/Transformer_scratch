def _get_next_assessment_section(self, assessment_section_id):
        """Gets the next section following section_id.

        Assumes that section list exists in taken and section_id is in section list.
        Assumes that Section parts only exist as children of Assessments

        """
        if self._my_map['sections'][-1] == str(assessment_section_id):
            # section_id represents the last seen section
            section = self._get_assessment_section(assessment_section_id)
            next_part_id, level = get_next_part_id(section._assessment_part_id,
                                                   runtime=self._runtime,
                                                   proxy=self._proxy,
                                                   sequestered=True)  # Raises IllegalState
            next_section = self._create_section(next_part_id)
            self._my_map['sections'].append(str(next_section.get_id()))
            self._save()
            return next_section
        else:
            return self._get_assessment_section(
                Id(self._my_map['sections'][self._my_map['sections'].index(str(assessment_section_id)) + 1]))