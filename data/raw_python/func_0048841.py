def delete_assessment_part(self, assessment_part_id):
        """Removes an asessment part and all mapped items.

        arg:    assessment_part_id (osid.id.Id): the ``Id`` of the
                ``AssessmentPart``
        raise:  NotFound - ``assessment_part_id`` not found
        raise:  NullArgument - ``assessment_part_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Should be implemented from template for
        # osid.learning.ObjectiveAdminSession.delete_objective_template
        # but need to handle magic part delete ...

        if not isinstance(assessment_part_id, ABCId):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        collection = JSONClientValidated('assessment_authoring',
                                         collection='AssessmentPart',
                                         runtime=self._runtime)
        if collection.find({'assessmentPartId': str(assessment_part_id)}).count() != 0:
            raise errors.IllegalState('there are still AssessmentParts associated with this AssessmentPart')

        collection = JSONClientValidated('assessment_authoring',
                                         collection='AssessmentPart',
                                         runtime=self._runtime)
        try:
            apls = get_assessment_part_lookup_session(runtime=self._runtime,
                                                      proxy=self._proxy)
            apls.use_unsequestered_assessment_part_view()
            apls.use_federated_bank_view()
            part = apls.get_assessment_part(assessment_part_id)
            part.delete()
        except AttributeError:
            collection.delete_one({'_id': ObjectId(assessment_part_id.get_identifier())})