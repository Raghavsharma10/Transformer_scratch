def get_banks_by_assessment_part(self, assessment_part_id):
        """Gets the ``Banks`` mapped to an ``AssessmentPart``.

        arg:    assessment_part_id (osid.id.Id): ``Id`` of an
                ``AssessmentPart``
        return: (osid.assessment.BankList) - list of banks
        raise:  NotFound - ``assessment_part_id`` is not found
        raise:  NullArgument - ``assessment_part_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        mgr = self._get_provider_manager('ASSESSMENT', local=True)
        lookup_session = mgr.get_bank_lookup_session(proxy=self._proxy)
        return lookup_session.get_banks_by_ids(
            self.get_bank_ids_by_assessment_part(assessment_part_id))