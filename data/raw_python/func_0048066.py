def get_assessment_offered_ids_by_banks(self, bank_ids):
        """Gets the list of ``AssessmentOffered Ids`` corresponding to a list of ``Banks``.

        arg:    bank_ids (osid.id.IdList): list of bank ``Ids``
        return: (osid.id.IdList) - list of bank ``Ids``
        raise:  NullArgument - ``bank_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resource_ids_by_bins
        id_list = []
        for assessment_offered in self.get_assessments_offered_by_banks(bank_ids):
            id_list.append(assessment_offered.get_id())
        return IdList(id_list)