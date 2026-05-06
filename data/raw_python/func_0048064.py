def get_assessment_offered_ids_by_bank(self, bank_id):
        """Gets the list of ``AssessmentOffered``  ``Ids`` associated with a ``Bank``.

        arg:    bank_id (osid.id.Id): ``Id`` of the ``Bank``
        return: (osid.id.IdList) - list of related assessment offered
                ``Ids``
        raise:  NotFound - ``bank_id`` is not found
        raise:  NullArgument - ``bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resource_ids_by_bin
        id_list = []
        for assessment_offered in self.get_assessments_offered_by_bank(bank_id):
            id_list.append(assessment_offered.get_id())
        return IdList(id_list)