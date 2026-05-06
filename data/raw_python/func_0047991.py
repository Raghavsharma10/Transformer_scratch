def get_first_unanswered_question(self, assessment_section_id):
        """Gets the first unanswered question in this assesment section.

        arg:    assessment_section_id (osid.id.Id): ``Id`` of the
                ``AssessmentSection``
        return: (osid.assessment.Question) - the first unanswered
                question
        raise:  IllegalState - ``has_unanswered_questions()`` is
                ``false``
        raise:  NotFound - ``assessment_section_id`` is not found
        raise:  NullArgument - ``assessment_section_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        questions = self.get_unanswered_questions(assessment_section_id)
        if not questions.available():
            raise errors.IllegalState('There are no more unanswered questions available')
        return questions.next()