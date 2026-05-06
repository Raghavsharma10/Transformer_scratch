def get_objective_ids_by_objective_banks(self, objective_bank_ids):
        """Gets the list of ``Objective Ids`` corresponding to a list of ``ObjectiveBanks``.

        arg:    objective_bank_ids (osid.id.IdList): list of objective
                bank ``Ids``
        return: (osid.id.IdList) - list of objective ``Ids``
        raise:  NullArgument - ``objective_bank_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resource_ids_by_bins
        id_list = []
        for objective in self.get_objectives_by_objective_banks(objective_bank_ids):
            id_list.append(objective.get_id())
        return IdList(id_list)