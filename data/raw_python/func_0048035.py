def get_assessments(self):
        """Gets all ``Assessments``.

        In plenary mode, the returned list contains all known
        assessments or an error results. Otherwise, the returned list
        may contain only those assessments that are accessible through
        this session.

        return: (osid.assessment.AssessmentList) - a list of
                ``Assessments``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('assessment',
                                         collection='Assessment',
                                         runtime=self._runtime)
        result = collection.find(self._view_filter()).sort('_id', DESCENDING)
        return objects.AssessmentList(result, runtime=self._runtime, proxy=self._proxy)