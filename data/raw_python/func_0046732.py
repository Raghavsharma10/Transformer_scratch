def get_gradebook_columns_by_genus_type(self, gradebook_column_genus_type):
        """Gets a ``GradebookColumnList`` corresponding to the given gradebook column genus ``Type`` which does not include gradebook columns of genus types derived from the specified ``Type``.

        In plenary mode, the returned list contains all known gradebook
        columns or an error results. Otherwise, the returned list may
        contain only those gradebook columns that are accessible through
        this session.

        arg:    gradebook_column_genus_type (osid.type.Type): a
                gradebook column genus type
        return: (osid.grading.GradebookColumnList) - the returned
                ``GradebookColumn`` list
        raise:  NullArgument - ``gradebook_column_genus_type`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources_by_genus_type
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('grading',
                                         collection='GradebookColumn',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'genusTypeId': str(gradebook_column_genus_type)},
                 **self._view_filter())).sort('_id', DESCENDING)
        return objects.GradebookColumnList(result, runtime=self._runtime, proxy=self._proxy)