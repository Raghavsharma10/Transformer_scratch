def remove_item(self, item_id, assessment_part_id):
        """Removes an ``Item`` from an ``AssessmentPartId``.

        arg:    item_id (osid.id.Id): ``Id`` of the ``Item``
        arg:    assessment_part_id (osid.id.Id): ``Id`` of the
                ``AssessmentPartId``
        raise:  NotFound - ``item_id``  ``not found in
                assessment_part_id``
        raise:  NullArgument - ``item_id`` or ``assessment_part_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization fauilure
        *compliance: mandatory -- This method must be implemented.*

        """
        if (not isinstance(assessment_part_id, ABCId) and
                assessment_part_id.get_identifier_namespace() != 'assessment_authoring.AssessmentPart'):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        assessment_part_map, collection = self._get_assessment_part_collection(assessment_part_id)
        try:
            assessment_part_map['itemIds'].remove(str(item_id))
        except (KeyError, ValueError):
            raise errors.NotFound()
        collection.save(assessment_part_map)