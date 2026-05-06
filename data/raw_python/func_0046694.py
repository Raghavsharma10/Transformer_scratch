def get_grade_systems_by_ids(self, grade_system_ids):
        """Gets a ``GradeSystemList`` corresponding to the given ``IdList``.

        In plenary mode, the returned list contains all of the systems
        specified in the ``Id`` list, in the order of the list,
        including duplicates, or an error results if an ``Id`` in the
        supplied list is not found or inaccessible. Otherwise,
        inaccessible ``GradeSystems`` may be omitted from the list and
        may present the elements in any order including returning a
        unique set.

        arg:    grade_system_ids (osid.id.IdList): the list of ``Ids``
                to retrieve
        return: (osid.grading.GradeSystemList) - the returned
                ``GradeSystem`` list
        raise:  NotFound - an ``Id was`` not found
        raise:  NullArgument - ``grade_system_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources_by_ids
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('grading',
                                         collection='GradeSystem',
                                         runtime=self._runtime)
        object_id_list = []
        for i in grade_system_ids:
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
        return objects.GradeSystemList(sorted_result, runtime=self._runtime, proxy=self._proxy)