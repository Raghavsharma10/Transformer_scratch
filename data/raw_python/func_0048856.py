def move_item_ahead(self, item_id, assessment_part_id, reference_id):
        """Reorders items in an assessment part by moving the specified item in front of a reference item.

        arg:    item_id (osid.id.Id): ``Id`` of the ``Item``
        arg:    assessment_part_id (osid.id.Id): ``Id`` of the
                ``AssessmentPartId``
        arg:    reference_id (osid.id.Id): ``Id`` of the reference
                ``Item``
        raise:  NotFound - ``item_id`` or ``reference_id``  ``not found
                in assessment_part_id``
        raise:  NullArgument - ``item_id, reference_id`` or
                ``assessment_part_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization fauilure
        *compliance: mandatory -- This method must be implemented.*

        """
        if (not isinstance(assessment_part_id, ABCId) and
                assessment_part_id.get_identifier_namespace() != 'assessment_authoring.AssessmentPart'):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        assessment_part_map, collection = self._get_assessment_part_collection(assessment_part_id)
        assessment_part_map['itemIds'] = move_id_ahead(item_id, reference_id, assessment_part_map['itemIds'])
        collection.save(assessment_part_map)