def get_objectives_by_objective_banks(self, objective_bank_ids):
        """Gets the list of ``Objectives`` corresponding to a list of ``ObjectiveBanks``.

        arg:    objective_bank_ids (osid.id.IdList): list of objective
                bank ``Ids``
        return: (osid.learning.ObjectiveList) - list of objectives
        raise:  NullArgument - ``objective_bank_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resources_by_bins
        objective_list = []
        for objective_bank_id in objective_bank_ids:
            objective_list += list(
                self.get_objectives_by_objective_bank(objective_bank_id))
        return objects.ObjectiveList(objective_list)