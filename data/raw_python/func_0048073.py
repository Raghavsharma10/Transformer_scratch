def get_assessments_taken_by_ids(self, assessment_taken_ids):
        """Gets an ``AssessmentTakenList`` corresponding to the given ``IdList``.

        In plenary mode, the returned list contains all of the
        assessments specified in the ``Id`` list, in the order of the
        list, including duplicates, or an error results if an ``Id`` in
        the supplied list is not found or inaccessible. Otherwise,
        inaccessible ``AssessmentTaken`` objects may be omitted from the
        list and may present the elements in any order including
        returning a unique set.

        arg:    assessment_taken_ids (osid.id.IdList): the list of
                ``Ids`` to retrieve
        return: (osid.assessment.AssessmentTakenList) - the returned
                ``AssessmentTaken list``
        raise:  NotFound - an ``Id was`` not found
        raise:  NullArgument - ``assessment_taken_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - assessment failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources_by_ids
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('assessment',
                                         collection='AssessmentTaken',
                                         runtime=self._runtime)
        object_id_list = []
        for i in assessment_taken_ids:
            object_id_list.append(ObjectId(self._get_id(i, 'assessment').get_identifier()))
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
        return objects.AssessmentTakenList(sorted_result, runtime=self._runtime, proxy=self._proxy)