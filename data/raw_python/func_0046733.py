def get_gradebook_columns(self):
        """Gets all gradebook columns.

        In plenary mode, the returned list contains all known gradebook
        columns or an error results. Otherwise, the returned list may
        contain only those gradebook columns that are accessible through
        this session.

        return: (osid.grading.GradebookColumnList) - a
                ``GradebookColumn``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('grading',
                                         collection='GradebookColumn',
                                         runtime=self._runtime)
        result = collection.find(self._view_filter()).sort('_id', DESCENDING)
        return objects.GradebookColumnList(result, runtime=self._runtime, proxy=self._proxy)