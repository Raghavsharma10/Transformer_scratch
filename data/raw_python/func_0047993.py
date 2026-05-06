def get_next_unanswered_question(self, assessment_section_id, item_id):
        """Gets the next unanswered question in this assesment section.

        arg:    assessment_section_id (osid.id.Id): ``Id`` of the
                ``AssessmentSection``
        arg:    item_id (osid.id.Id): ``Id`` of the ``Item``
        return: (osid.assessment.Question) - the next unanswered
                question
        raise:  IllegalState - ``has_next_unanswered_question()`` is
                ``false``
        raise:  NotFound - ``assessment_section_id or item_id is not
                found, or item_id not part of assessment_section_id``
        raise:  NullArgument - ``assessment_section_id or item_id is
                null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Or this could call through to get_next_question in the section
        questions = self.get_unanswered_questions(assessment_section_id)
        for question in questions:
            if question.get_id() == item_id:
                if questions.available():
                    return questions.next()
                else:
                    raise errors.IllegalState('No next unanswered question is available')
        raise errors.NotFound('item_id is not found in Section')