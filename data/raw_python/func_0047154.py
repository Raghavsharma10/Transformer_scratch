def get_objective_ids_by_objective_bank(self, objective_bank_id):
        """Gets the list of ``Objective``  ``Ids`` associated with an ``ObjectiveBank``.

        arg:    objective_bank_id (osid.id.Id): ``Id`` of the
                ``ObjectiveBank``
        return: (osid.id.IdList) - list of related objectives
        raise:  NotFound - ``objective_bank_id`` is not found
        raise:  NullArgument - ``objective_bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resource_ids_by_bin
        id_list = []
        for objective in self.get_objectives_by_objective_bank(objective_bank_id):
            id_list.append(objective.get_id())
        return IdList(id_list)