def get_assessment_parts(self):
        """Gets all ``AssessmentParts``.

        return: (osid.assessment.authoring.AssessmentPartList) - a list
                of ``AssessmentParts``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('assessment_authoring',
                                         collection='AssessmentPart',
                                         runtime=self._runtime)
        result = collection.find(self._view_filter()).sort('_id', DESCENDING)
        return objects.AssessmentPartList(result, runtime=self._runtime, proxy=self._proxy)