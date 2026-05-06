def get_grade_entries_by_ids(self, grade_entry_ids):
        """Gets a ``GradeEntryList`` corresponding to the given ``IdList``.

        arg:    grade_entry_ids (osid.id.IdList): the list of ``Ids`` to
                retrieve
        return: (osid.grading.GradeEntryList) - the returned
                ``GradeEntry`` list
        raise:  NotFound - an ``Id was`` not found
        raise:  NullArgument - ``grade_entry_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources_by_ids
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('grading',
                                         collection='GradeEntry',
                                         runtime=self._runtime)
        object_id_list = []
        for i in grade_entry_ids:
            object_id_list.append(ObjectId(self._get_id(i, 'grading').get_identifier()))
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
        return objects.GradeEntryList(sorted_result, runtime=self._runtime, proxy=self._proxy)