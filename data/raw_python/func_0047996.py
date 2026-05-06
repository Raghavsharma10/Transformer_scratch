def get_response(self, assessment_section_id, item_id):
        """Gets the submitted response to the associated item.

        arg:    assessment_section_id (osid.id.Id): ``Id`` of the
                ``AssessmentSection``
        arg:    item_id (osid.id.Id): ``Id`` of the ``Item``
        return: (osid.assessment.Response) - the response
        raise:  IllegalState - ``has_assessment_section_begun()`` is
                ``false or is_assessment_section_over()`` is ``true``
        raise:  NotFound - ``assessment_section_id or item_id is not
                found, or item_id not part of assessment_section_id``
        raise:  NullArgument - ``assessment_section_id or item_id is
                null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        return self.get_assessment_section(assessment_section_id).get_response(question_id=item_id)