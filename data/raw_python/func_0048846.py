def get_assessment_parts_by_banks(self, bank_ids):
        """Gets the list of assessment part corresponding to a list of ``Banks``.

        arg:    bank_ids (osid.id.IdList): list of bank ``Ids``
        return: (osid.assessment.authoring.AssessmentPartList) - list of
                assessment parts
        raise:  NullArgument - ``bank_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resources_by_bins
        assessment_part_list = []
        for bank_id in bank_ids:
            assessment_part_list += list(
                self.get_assessment_parts_by_bank(bank_id))
        return objects.AssessmentPartList(assessment_part_list)