def order_items(self, item_ids, assessment_part_id):
        """Reorders a set of items in an assessment part.

        arg:    item_ids (osid.id.Id[]): ``Ids`` for a set of ``Items``
        arg:    assessment_part_id (osid.id.Id): ``Id`` of the
                ``AssessmentPartId``
        raise:  NotFound - ``assessment_part_id`` not found or, an
                ``item_id`` not related to ``assessment_part_id``
        raise:  NullArgument - ``item_ids`` or ``agenda_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        if (not isinstance(assessment_part_id, ABCId) and
                assessment_part_id.get_identifier_namespace() != 'assessment_authoring.AssessmentPart'):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        assessment_part_map, collection = self._get_assessment_part_collection(assessment_part_id)
        assessment_part_map['itemIds'] = order_ids(item_ids, assessment_part_map['itemIds'])
        collection.save(assessment_part_map)