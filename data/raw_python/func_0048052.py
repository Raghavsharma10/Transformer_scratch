def add_item(self, assessment_id, item_id):
        """Adds an existing ``Item`` to an assessment.

        arg:    assessment_id (osid.id.Id): the ``Id`` of the
                ``Assessment``
        arg:    item_id (osid.id.Id): the ``Id`` of the ``Item``
        raise:  NotFound - ``assessment_id`` or ``item_id`` not found
        raise:  NullArgument - ``assessment_id`` or ``item_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        if assessment_id.get_identifier_namespace() != 'assessment.Assessment':
            raise errors.InvalidArgument
        self._part_item_design_session.add_item(item_id, self._get_first_part_id(assessment_id))