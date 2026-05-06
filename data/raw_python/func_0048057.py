def get_assessments_offered_by_genus_type(self, assessment_offered_genus_type):
        """Gets an ``AssessmentOfferedList`` corresponding to the given assessment offered genus ``Type`` which does not include assessments of types derived from the specified ``Type``.

        In plenary mode, the returned list contains all known
        assessments offered or an error results. Otherwise, the returned
        list may contain only those assessments offered that are
        accessible through this session.

        arg:    assessment_offered_genus_type (osid.type.Type): an
                assessment offered genus type
        return: (osid.assessment.AssessmentOfferedList) - the returned
                ``AssessmentOffered`` list
        raise:  NullArgument - ``assessment_offered_genus_type`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources_by_genus_type
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('assessment',
                                         collection='AssessmentOffered',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'genusTypeId': str(assessment_offered_genus_type)},
                 **self._view_filter())).sort('_id', DESCENDING)
        return objects.AssessmentOfferedList(result, runtime=self._runtime, proxy=self._proxy)