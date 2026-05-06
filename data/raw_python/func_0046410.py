def _get_previous_assessment_section(self, assessment_section_id):
        """Gets the previous section before section_id.

        Assumes that section list exists in taken and section_id is in section list.
        Assumes that Section parts only exist as children of Assessments

        """
        if self._my_map['sections'][0] == str(assessment_section_id):
            raise errors.IllegalState('already at the first section')
        else:
            return self._get_assessment_section(
                Id(self._my_map['sections'][self._my_map['sections'].index(str(assessment_section_id)) - 1]))