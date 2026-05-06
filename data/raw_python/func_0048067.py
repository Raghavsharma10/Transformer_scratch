def get_assessments_offered_by_banks(self, bank_ids):
        """Gets the list of ``AssessmentOffered`` objects corresponding to a list of ``Banks``.

        arg:    bank_ids (osid.id.IdList): list of bank ``Ids``
        return: (osid.assessment.AssessmentOfferedList) - list of
                assessments offered
        raise:  NullArgument - ``bank_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resources_by_bins
        assessment_offered_list = []
        for bank_id in bank_ids:
            assessment_offered_list += list(
                self.get_assessments_offered_by_bank(bank_id))
        return objects.AssessmentOfferedList(assessment_offered_list)