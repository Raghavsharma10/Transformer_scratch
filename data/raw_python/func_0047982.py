def get_incomplete_assessment_sections(self, assessment_taken_id):
        """Gets the incomplete assessment sections of this assessment.

        arg:    assessment_taken_id (osid.id.Id): ``Id`` of the
                ``AssessmentTaken``
        return: (osid.assessment.AssessmentSectionList) - the list of
                incomplete assessment sections
        raise:  IllegalState - ``has_assessment_begun()`` is ``false``
        raise:  NotFound - ``assessment_taken_id`` is not found
        raise:  NullArgument - ``assessment_taken_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        section_list = []
        for section in self.get_assessment_sections(assessment_taken_id):
            if not section.is_complete():
                section_list.append(section)
        return objects.AssessmentSectionList(section_list, runtime=self._runtime, proxy=self._proxy)