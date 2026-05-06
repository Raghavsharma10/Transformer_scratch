def get_answers(self, assessment_section_id, item_id):
        """Gets the acceptable answers to the associated item.

        arg:    assessment_section_id (osid.id.Id): ``Id`` of the
                ``AssessmentSection``
        arg:    item_id (osid.id.Id): ``Id`` of the ``Item``
        return: (osid.assessment.AnswerList) - the answers
        raise:  IllegalState - ``is_answer_available()`` is ``false``
        raise:  NotFound - ``assessment_section_id or item_id is not
                found, or item_id not part of assessment_section_id``
        raise:  NullArgument - ``assessment_section_id or item_id is
                null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.is_answer_available(assessment_section_id, item_id):
            return self.get_assessment_section(assessment_section_id).get_answers(question_id=item_id)
        raise errors.IllegalState()