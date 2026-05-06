def get_banks_by_assessment_offered(self, assessment_offered_id):
        """Gets the list of ``Banks`` mapped to an ``AssessmentOffered``.

        arg:    assessment_offered_id (osid.id.Id): ``Id`` of an
                ``AssessmentOffered``
        return: (osid.assessment.BankList) - list of banks
        raise:  NotFound - ``assessment_offered_id`` is not found
        raise:  NullArgument - ``assessment_offered_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_bins_by_resource
        mgr = self._get_provider_manager('ASSESSMENT', local=True)
        lookup_session = mgr.get_bank_lookup_session(proxy=self._proxy)
        return lookup_session.get_banks_by_ids(
            self.get_bank_ids_by_assessment_offered(assessment_offered_id))