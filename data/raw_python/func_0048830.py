def get_assessment_parts_by_ids(self, assessment_part_ids):
        """Gets an ``AssessmentPartList`` corresponding to the given ``IdList``.

        arg:    assessment_part_ids (osid.id.IdList): the list of
                ``Ids`` to retrieve
        return: (osid.assessment.authoring.AssessmentPartList) - the
                returned ``AssessmentPart`` list
        raise:  NotFound - an ``Id was`` not found
        raise:  NullArgument - ``assessment_part_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources_by_ids
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('assessment_authoring',
                                         collection='AssessmentPart',
                                         runtime=self._runtime)
        object_id_list = []
        for i in assessment_part_ids:
            object_id_list.append(ObjectId(self._get_id(i, 'assessment_authoring').get_identifier()))
        result = collection.find(
            dict({'_id': {'$in': object_id_list}},
                 **self._view_filter()))
        result = list(result)
        sorted_result = []
        for object_id in object_id_list:
            for object_map in result:
                if object_map['_id'] == object_id:
                    sorted_result.append(object_map)
                    break
        return objects.AssessmentPartList(sorted_result, runtime=self._runtime, proxy=self._proxy)