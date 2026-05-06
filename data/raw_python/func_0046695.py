def get_grade_systems_by_genus_type(self, grade_system_genus_type):
        """Gets a ``GradeSystemList`` corresponding to the given grade system genus ``Type`` which does not include systems of genus types derived from the specified ``Type``.

        In plenary mode, the returned list contains all known systems or
        an error results. Otherwise, the returned list may contain only
        those systems that are accessible through this session.

        arg:    grade_system_genus_type (osid.type.Type): a grade system
                genus type
        return: (osid.grading.GradeSystemList) - the returned
                ``GradeSystem`` list
        raise:  NullArgument - ``grade_system_genus_type`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources_by_genus_type
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('grading',
                                         collection='GradeSystem',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'genusTypeId': str(grade_system_genus_type)},
                 **self._view_filter())).sort('_id', DESCENDING)
        return objects.GradeSystemList(result, runtime=self._runtime, proxy=self._proxy)