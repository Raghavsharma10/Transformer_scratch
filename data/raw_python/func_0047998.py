def finish_assessment_section(self, assessment_section_id):
        """Indicates an assessment section is complete.

        Finished sections may or may not allow new or updated responses.

        arg:    assessment_section_id (osid.id.Id): ``Id`` of the
                ``AssessmentSection``
        raise:  IllegalState - ``has_assessment_section_begun()`` is
                ``false or is_assessment_section_over()`` is ``true``
        raise:  NotFound - ``assessment_section_id`` is not found
        raise:  NullArgument - ``assessment_section_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        if (not self.has_assessment_section_begun(assessment_section_id) or
                self.is_assessment_section_over(assessment_section_id)):
            raise errors.IllegalState()
        self.get_assessment_section(assessment_section_id).finish()