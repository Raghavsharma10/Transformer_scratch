def get_assessment_offered(self, assessment_offered_id):
        """Gets the ``AssessmentOffered`` specified by its ``Id``.

        In plenary mode, the exact ``Id`` is found or a ``NotFound``
        results. Otherwise, the returned ``AssessmentOffered`` may have
        a different ``Id`` than requested, such as the case where a
        duplicate ``Id`` was assigned to an ``AssessmentOffered`` and
        retained for compatibility.

        arg:    assessment_offered_id (osid.id.Id): ``Id`` of the
                ``AssessmentOffered``
        return: (osid.assessment.AssessmentOffered) - the assessment
                offered
        raise:  NotFound - ``assessment_offered_id`` not found
        raise:  NullArgument - ``assessment_offered_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method is must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resource
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('assessment',
                                         collection='AssessmentOffered',
                                         runtime=self._runtime)
        result = collection.find_one(
            dict({'_id': ObjectId(self._get_id(assessment_offered_id, 'assessment').get_identifier())},
                 **self._view_filter()))
        return objects.AssessmentOffered(osid_object_map=result, runtime=self._runtime, proxy=self._proxy)