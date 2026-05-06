def get_assessments_taken_by_banks(self, bank_ids):
        """Gets the list of ``AssessmentTaken`` objects corresponding to a list of ``Banks``.

        arg:    bank_ids (osid.id.IdList): list of bank ``Ids``
        return: (osid.assessment.AssessmentTakenList) - list of
                assessments taken
        raise:  NullArgument - ``bank_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resources_by_bins
        assessment_taken_list = []
        for bank_id in bank_ids:
            assessment_taken_list += list(
                self.get_assessments_taken_by_bank(bank_id))
        return objects.AssessmentTakenList(assessment_taken_list)