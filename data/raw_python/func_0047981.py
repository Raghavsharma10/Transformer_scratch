def get_assessment_section(self, assessment_section_id):
        """Gets an assessemnts section by ``Id``.

        arg:    assessment_section_id (osid.id.Id): ``Id`` of the
                ``AssessmentSection``
        return: (osid.assessment.AssessmentSection) - the assessment
                section
        raise:  IllegalState - ``has_assessment_begun()`` is ``false``
        raise:  NotFound - ``assessment_section_id`` is not found
        raise:  NullArgument - ``assessment_section_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        return get_section_util(assessment_section_id, runtime=self._runtime, proxy=self._proxy)