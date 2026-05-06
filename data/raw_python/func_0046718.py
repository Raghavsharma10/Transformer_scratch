def get_grade_entries_by_genus_type(self, grade_entry_genus_type):
        """Gets a ``GradeEntryList`` corresponding to the given grade entry genus ``Type`` which does not include grade entries of genus types derived from the specified ``Type``.

        arg:    grade_entry_genus_type (osid.type.Type): a grade entry
                genus type
        return: (osid.grading.GradeEntryList) - the returned
                ``GradeEntry`` list
        raise:  NullArgument - ``grade_entry_genus_type`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources_by_genus_type
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('grading',
                                         collection='GradeEntry',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'genusTypeId': str(grade_entry_genus_type)},
                 **self._view_filter())).sort('_id', DESCENDING)
        return objects.GradeEntryList(result, runtime=self._runtime, proxy=self._proxy)