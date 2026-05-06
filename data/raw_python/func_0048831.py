def get_assessment_parts_by_genus_type(self, assessment_part_genus_type):
        """Gets an ``AssessmentPartList`` corresponding to the given assessment part genus ``Type`` which does not include assessment parts of types derived from the specified ``Type``.

        arg:    assessment_part_genus_type (osid.type.Type): an
                assessment part genus type
        return: (osid.assessment.authoring.AssessmentPartList) - the
                returned ``AssessmentPart`` list
        raise:  NullArgument - ``assessment_part_genus_type`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources_by_genus_type
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('assessment_authoring',
                                         collection='AssessmentPart',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'genusTypeId': str(assessment_part_genus_type)},
                 **self._view_filter())).sort('_id', DESCENDING)
        return objects.AssessmentPartList(result, runtime=self._runtime, proxy=self._proxy)