def get_assessment(self, assessment_id):
        """Gets the ``Assessment`` specified by its ``Id``.

        In plenary mode, the exact ``Id`` is found or a ``NotFound``
        results. Otherwise, the returned ``Assessment`` may have a
        different ``Id`` than requested, such as the case where a
        duplicate ``Id`` was assigned to a ``Assessment`` and retained
        for compatibility.

        arg:    assessment_id (osid.id.Id): ``Id`` of the ``Assessment``
        return: (osid.assessment.Assessment) - the assessment
        raise:  NotFound - ``assessment_id`` not found
        raise:  NullArgument - ``assessment_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method is must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resource
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('assessment',
                                         collection='Assessment',
                                         runtime=self._runtime)
        result = collection.find_one(
            dict({'_id': ObjectId(self._get_id(assessment_id, 'assessment').get_identifier())},
                 **self._view_filter()))
        return objects.Assessment(osid_object_map=result, runtime=self._runtime, proxy=self._proxy)