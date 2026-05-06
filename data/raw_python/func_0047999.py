def is_answer_available(self, assessment_section_id, item_id):
        """Tests if an answer is available for the given item.

        arg:    assessment_section_id (osid.id.Id): ``Id`` of the
                ``AssessmentSection``
        arg:    item_id (osid.id.Id): ``Id`` of the ``Item``
        return: (boolean) - ``true`` if an answer are available,
                ``false`` otherwise
        raise:  NotFound - ``assessment_section_id or item_id is not
                found, or item_id not part of assessment_section_id``
        raise:  NullArgument - ``assessment_section_id or item_id is
                null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Note: we need more settings elsewhere to indicate answer available conditions
        # This makes the simple assumption that answers are available only when
        # a response has been submitted for an Item.
        try:
            response = self.get_response(assessment_section_id, item_id)
            # need to invoke something like .object_map before
            # a "null" response throws IllegalState
            response.object_map
        except errors.IllegalState:
            return False
        else:
            return True