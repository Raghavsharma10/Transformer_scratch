def get_grade_system(self, grade_system_id):
        """Gets the ``GradeSystem`` specified by its ``Id``.

        In plenary mode, the exact ``Id`` is found or a ``NotFound``
        results. Otherwise, the returned ``GradeSystem`` may have a
        different ``Id`` than requested, such as the case where a
        duplicate ``Id`` was assigned to a ``GradeSystem`` and retained
        for compatibility.

        arg:    grade_system_id (osid.id.Id): ``Id`` of the
                ``GradeSystem``
        return: (osid.grading.GradeSystem) - the grade system
        raise:  NotFound - ``grade_system_id`` not found
        raise:  NullArgument - ``grade_system_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method is must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resource
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('grading',
                                         collection='GradeSystem',
                                         runtime=self._runtime)
        result = collection.find_one(
            dict({'_id': ObjectId(self._get_id(grade_system_id, 'grading').get_identifier())},
                 **self._view_filter()))
        return objects.GradeSystem(osid_object_map=result, runtime=self._runtime, proxy=self._proxy)