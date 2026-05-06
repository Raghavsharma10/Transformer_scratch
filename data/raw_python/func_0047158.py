def get_objective_bank_ids_by_objective(self, objective_id):
        """Gets the list of ``ObjectiveBank``  ``Ids`` mapped to an ``Objective``.

        arg:    objective_id (osid.id.Id): ``Id`` of an ``Objective``
        return: (osid.id.IdList) - list of objective bank ``Ids``
        raise:  NotFound - ``objective_id`` is not found
        raise:  NullArgument - ``objective_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_bin_ids_by_resource
        mgr = self._get_provider_manager('LEARNING', local=True)
        lookup_session = mgr.get_objective_lookup_session(proxy=self._proxy)
        lookup_session.use_federated_objective_bank_view()
        objective = lookup_session.get_objective(objective_id)
        id_list = []
        for idstr in objective._my_map['assignedObjectiveBankIds']:
            id_list.append(Id(idstr))
        return IdList(id_list)