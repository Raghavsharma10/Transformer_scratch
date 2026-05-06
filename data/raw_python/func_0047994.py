def has_previous_unanswered_question(self, assessment_section_id, item_id):
        """Tests if there is a previous unanswered question preceeding the given question ``Id``.

        arg:    assessment_section_id (osid.id.Id): ``Id`` of the
                ``AssessmentSection``
        arg:    item_id (osid.id.Id): ``Id`` of the ``Item``
        return: (boolean) - ``true`` if there is a previous unanswered
                question, ``false`` otherwise
        raise:  IllegalState - ``has_assessment_section_begun() is false
                or is_assessment_section_over() is true``
        raise:  NotFound - ``assessment_section_id or item_id is not
                found, or item_id not part of assessment_section_id``
        raise:  NullArgument - ``assessment_section_id or item_id is
                null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # There's probably a more efficient way to implement this:
        try:
            self.get_previous_unanswered_question(assessment_section_id, item_id)
        except errors.IllegalState:
            return False
        else:
            return True