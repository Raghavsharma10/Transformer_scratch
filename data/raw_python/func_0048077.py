def get_assessments_taken_for_taker_and_assessment_offered(self, resource_id, assessment_offered_id):
        """Gets an ``AssessmentTakenList`` for the given resource and assessment offered.

        In plenary mode, the returned list contains all known
        assessments taken or an error results. Otherwise, the returned
        list may contain only those assessments taken that are
        accessible through this session.

        arg:    resource_id (osid.id.Id): ``Id`` of a ``Resource``
        arg:    assessment_offered_id (osid.id.Id): ``Id`` of an
                ``AssessmentOffered``
        return: (osid.assessment.AssessmentTakenList) - the returned
                ``AssessmentTaken`` list
        raise:  NullArgument - ``resource_id`` or
                ``assessmen_offeredt_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('assessment',
                                         collection='AssessmentTaken',
                                         runtime=self._runtime)

        am = self._get_provider_manager('ASSESSMENT')
        aols = am.get_assessment_offered_lookup_session(proxy=self._proxy)
        aols.use_federated_bank_view()
        offered = aols.get_assessment_offered(assessment_offered_id)
        try:
            deadline = offered.get_deadline()
            nowutc = DateTime.utcnow()
            if nowutc > deadline:
                raise errors.PermissionDenied('you are passed the deadline for the offered')
        except errors.IllegalState:
            # no deadline set
            pass

        result = collection.find(
            dict({'assessmentOfferedId': str(assessment_offered_id),
                  'takingAgentId': str(resource_id)},
                 **self._view_filter())).sort('_id', DESCENDING)
        return objects.AssessmentTakenList(result, runtime=self._runtime, proxy=self._proxy)