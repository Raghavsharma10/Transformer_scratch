def order_items(self, item_ids, assessment_id):
        """Sequences existing items in an assessment.

        arg:    item_ids (osid.id.Id[]): the ``Id`` of the ``Items``
        arg:    assessment_id (osid.id.Id): the ``Id`` of the
                ``Assessment``
        raise:  NotFound - ``assessment_id`` is not found or an
                ``item_id`` is not on ``assessment_id``
        raise:  NullArgument - ``assessment_id`` or ``item_ids`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        if assessment_id.get_identifier_namespace() != 'assessment.Assessment':
            raise errors.InvalidArgument
        self._part_item_design_session.order_items(item_ids, self._get_first_part_id(assessment_id))