def get_banks_by_assessment_taken(self, assessment_taken_id):
        """Gets the list of ``Banks`` mapped to an ``AssessmentTaken``.

        arg:    assessment_taken_id (osid.id.Id): ``Id`` of an
                ``AssessmentTaken``
        return: (osid.assessment.BankList) - list of banks
        raise:  NotFound - ``assessment_taken_id`` is not found
        raise:  NullArgument - ``assessment_taken_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_bins_by_resource
        mgr = self._get_provider_manager('ASSESSMENT', local=True)
        lookup_session = mgr.get_bank_lookup_session(proxy=self._proxy)
        return lookup_session.get_banks_by_ids(
            self.get_bank_ids_by_assessment_taken(assessment_taken_id))