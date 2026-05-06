def get_activities_by_objective_banks(self, objective_bank_ids):
        """Gets the list of ``Activities`` corresponding to a list of ``ObjectiveBanks``.

        arg:    objective_bank_ids (osid.id.IdList): list of objective
                bank ``Ids``
        return: (osid.learning.ActivityList) - list of activities
        raise:  NullArgument - ``objective_bank_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resources_by_bins
        activity_list = []
        for objective_bank_id in objective_bank_ids:
            activity_list += list(
                self.get_activities_by_objective_bank(objective_bank_id))
        return objects.ActivityList(activity_list)