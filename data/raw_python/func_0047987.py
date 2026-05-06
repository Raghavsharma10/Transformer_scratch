def get_previous_question(self, assessment_section_id, item_id):
        """Gets the previous question in this assesment section.

        arg:    assessment_section_id (osid.id.Id): ``Id`` of the
                ``AssessmentSection``
        arg:    item_id (osid.id.Id): ``Id`` of the ``Item``
        return: (osid.assessment.Question) - the previous question
        raise:  IllegalState - ``has_previous_question()`` is ``false``
        raise:  NotFound - ``assessment_section_id`` or ``item_id`` is
                not found, or ``item_id`` not part of
                ``assessment_section_id``
        raise:  NullArgument - ``assessment_section_id`` or ``item_id``
                is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        return self.get_assessment_section(assessment_section_id).get_next_question(question_id=item_id, reverse=True)