def get_grade_systems(self):
        """Gets all ``GradeSystems``.

        In plenary mode, the returned list contains all known grade
        systems or an error results. Otherwise, the returned list may
        contain only those grade systems that are accessible through
        this session.

        return: (osid.grading.GradeSystemList) - a ``GradeSystemList``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('grading',
                                         collection='GradeSystem',
                                         runtime=self._runtime)
        result = collection.find(self._view_filter()).sort('_id', DESCENDING)
        return objects.GradeSystemList(result, runtime=self._runtime, proxy=self._proxy)