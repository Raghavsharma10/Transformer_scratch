def get_assessment_part_ids_by_banks(self, bank_ids):
        """Gets the list of ``AssessmentPart Ids`` corresponding to a list of ``Banks``.

        arg:    bank_ids (osid.id.IdList): list of bank ``Ids``
        return: (osid.id.IdList) - list of assessment part ``Ids``
        raise:  NullArgument - ``bank_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resource_ids_by_bins
        id_list = []
        for assessment_part in self.get_assessment_parts_by_banks(bank_ids):
            id_list.append(assessment_part.get_id())
        return IdList(id_list)