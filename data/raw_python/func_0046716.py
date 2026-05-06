def get_grade_entry(self, grade_entry_id):
        """Gets the ``GradeEntry`` specified by its ``Id``.

        arg:    grade_entry_id (osid.id.Id): ``Id`` of the
                ``GradeEntry``
        return: (osid.grading.GradeEntry) - the grade entry
        raise:  NotFound - ``grade_entry_id`` not found
        raise:  NullArgument - ``grade_entry_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method is must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resource
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('grading',
                                         collection='GradeEntry',
                                         runtime=self._runtime)
        result = collection.find_one(
            dict({'_id': ObjectId(self._get_id(grade_entry_id, 'grading').get_identifier())},
                 **self._view_filter()))
        return objects.GradeEntry(osid_object_map=result, runtime=self._runtime, proxy=self._proxy)