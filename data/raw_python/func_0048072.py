def get_assessment_taken(self, assessment_taken_id):
        """Gets the ``AssessmentTaken`` specified by its ``Id``.

        In plenary mode, the exact ``Id`` is found or a ``NotFound``
        results. Otherwise, the returned ``AssessmentTaken`` may have a
        different ``Id`` than requested, such as the case where a
        duplicate ``Id`` was assigned to an ``AssessmentTaken`` and
        retained for compatibility.

        arg:    assessment_taken_id (osid.id.Id): ``Id`` of the
                ``AssessmentTaken``
        return: (osid.assessment.AssessmentTaken) - the assessment taken
        raise:  NotFound - ``assessment_taken_id`` not found
        raise:  NullArgument - ``assessment_taken_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method is must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resource
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('assessment',
                                         collection='AssessmentTaken',
                                         runtime=self._runtime)
        result = collection.find_one(
            dict({'_id': ObjectId(self._get_id(assessment_taken_id, 'assessment').get_identifier())},
                 **self._view_filter()))
        return objects.AssessmentTaken(osid_object_map=result, runtime=self._runtime, proxy=self._proxy)