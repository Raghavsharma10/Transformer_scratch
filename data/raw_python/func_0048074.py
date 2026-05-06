def get_assessments_taken_by_genus_type(self, assessment_taken_genus_type):
        """Gets an ``AssessmentTakenList`` corresponding to the given assessment taken genus ``Type`` which does not include assessments of types derived from the specified ``Type``.

        In plenary mode, the returned list contains all known
        assessments taken or an error results. Otherwise, the returned
        list may contain only those assessments taken that are
        accessible through this session.

        arg:    assessment_taken_genus_type (osid.type.Type): an
                assessment taken genus type
        return: (osid.assessment.AssessmentTakenList) - the returned
                ``AssessmentTaken list``
        raise:  NullArgument - ``assessment_taken_genus_type`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources_by_genus_type
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('assessment',
                                         collection='AssessmentTaken',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'genusTypeId': str(assessment_taken_genus_type)},
                 **self._view_filter())).sort('_id', DESCENDING)
        return objects.AssessmentTakenList(result, runtime=self._runtime, proxy=self._proxy)