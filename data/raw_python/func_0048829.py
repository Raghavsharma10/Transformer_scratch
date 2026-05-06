def get_assessment_part(self, assessment_part_id):
        """Gets the ``AssessmentPart`` specified by its ``Id``.

        arg:    assessment_part_id (osid.id.Id): the ``Id`` of the
                ``AssessmentPart`` to retrieve
        return: (osid.assessment.authoring.AssessmentPart) - the
                returned ``AssessmentPart``
        raise:  NotFound - no ``AssessmentPart`` found with the given
                ``Id``
        raise:  NullArgument - ``assessment_part_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resource
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('assessment_authoring',
                                         collection='AssessmentPart',
                                         runtime=self._runtime)
        result = collection.find_one(
            dict({'_id': ObjectId(self._get_id(assessment_part_id, 'assessment_authoring').get_identifier())},
                 **self._view_filter()))
        return objects.AssessmentPart(osid_object_map=result, runtime=self._runtime, proxy=self._proxy)