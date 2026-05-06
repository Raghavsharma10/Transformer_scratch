def get_gradebook_column(self, gradebook_column_id):
        """Gets the ``GradebookColumn`` specified by its ``Id``.

        In plenary mode, the exact ``Id`` is found or a ``NotFound``
        results. Otherwise, the returned ``GradebookColumn`` may have a
        different ``Id`` than requested, such as the case where a
        duplicate ``Id`` was assigned to a ``GradebookColumn`` and
        retained for compatibility.

        arg:    gradebook_column_id (osid.id.Id): ``Id`` of the
                ``GradebookColumn``
        return: (osid.grading.GradebookColumn) - the gradebook column
        raise:  NotFound - ``gradebook_column_id`` not found
        raise:  NullArgument - ``gradebook_column_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method is must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resource
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('grading',
                                         collection='GradebookColumn',
                                         runtime=self._runtime)
        result = collection.find_one(
            dict({'_id': ObjectId(self._get_id(gradebook_column_id, 'grading').get_identifier())},
                 **self._view_filter()))
        return objects.GradebookColumn(osid_object_map=result, runtime=self._runtime, proxy=self._proxy)