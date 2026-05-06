def get_items(self, assessment_id):
        """Gets the items in sequence from an assessment.

        arg:    assessment_id (osid.id.Id): the ``Id`` of the
                ``Assessment``
        return: (osid.assessment.ItemList) - list of items
        raise:  NotFound - ``assessmentid`` not found
        raise:  NullArgument - ``assessment_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        if assessment_id.get_identifier_namespace() != 'assessment.Assessment':
            raise errors.InvalidArgument
        return self._part_item_session.get_assessment_part_items(self._get_first_part_id(assessment_id))