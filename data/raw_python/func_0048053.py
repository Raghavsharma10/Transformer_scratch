def move_item(self, assessment_id, item_id, preceeding_item_id):
        """Moves an existing item to follow another item in an assessment.

        arg:    assessment_id (osid.id.Id): the ``Id`` of the
                ``Assessment``
        arg:    item_id (osid.id.Id): the ``Id`` of an ``Item``
        arg:    preceeding_item_id (osid.id.Id): the ``Id`` of a
                preceeding ``Item`` in the sequence
        raise:  NotFound - ``assessment_id`` is not found, or
                ``item_id`` or ``preceeding_item_id`` not on
                ``assessment_id``
        raise:  NullArgument - ``assessment_id, item_id`` or
                ``preceeding_item_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        if assessment_id.get_identifier_namespace() != 'assessment.Assessment':
            raise errors.InvalidArgument
        self._part_item_design_session.move_item_behind(item_id, self._get_first_part_id(assessment_id), preceeding_item_id)