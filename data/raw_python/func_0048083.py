def delete_assessment_taken(self, assessment_taken_id):
        """Deletes an ``AssessmentTaken``.

        arg:    assessment_taken_id (osid.id.Id): the ``Id`` of the
                ``AssessmentTaken`` to remove
        raise:  NotFound - ``assessment_taken_id`` not found
        raise:  NullArgument - ``assessment_taken_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.delete_resource_template
        collection = JSONClientValidated('assessment',
                                         collection='AssessmentTaken',
                                         runtime=self._runtime)
        if not isinstance(assessment_taken_id, ABCId):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        assessment_taken_map = collection.find_one(
            dict({'_id': ObjectId(assessment_taken_id.get_identifier())},
                 **self._view_filter()))

        objects.AssessmentTaken(osid_object_map=assessment_taken_map, runtime=self._runtime, proxy=self._proxy)._delete()
        collection.delete_one({'_id': ObjectId(assessment_taken_id.get_identifier())})