def clear_response(self, assessment_section_id, item_id):
        """Clears the response to an item The item appears as unanswered.

        If no response exists, the method simply returns.

        arg:    assessment_section_id (osid.id.Id): ``Id`` of the
                ``AssessmentSection``
        arg:    item_id (osid.id.Id): ``Id`` of the ``Item``
        raise:  IllegalState - ``has_assessment_section_begun() is false
                or is_assessment_section_over() is true``
        raise:  NotFound - ``assessment_section_id or item_id is not
                found, or item_id not part of assessment_section_id``
        raise:  NullArgument - ``assessment_section_id or item_id is
                null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        if (not self.has_assessment_section_begun(assessment_section_id) or
                self.is_assessment_section_over(assessment_section_id)):
            raise errors.IllegalState()
        # Should probably check to see if responses can be cleared, but how?
        self.get_assessment_section(assessment_section_id).submit_response(item_id, None)