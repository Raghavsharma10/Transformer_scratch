def get_assessments_by_banks(self, bank_ids):
        """Gets the list of ``Assessments`` corresponding to a list of ``Banks``.

        arg:    bank_ids (osid.id.IdList): list of bank ``Ids``
        return: (osid.assessment.AssessmentList) - list of assessments
        raise:  NullArgument - ``bank_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resources_by_bins
        assessment_list = []
        for bank_id in bank_ids:
            assessment_list += list(
                self.get_assessments_by_bank(bank_id))
        return objects.AssessmentList(assessment_list)