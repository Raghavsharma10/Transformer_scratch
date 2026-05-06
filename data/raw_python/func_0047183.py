def get_objective_bank_ids_by_activity(self, activity_id):
        """Gets the list of ``ObjectiveBank Ids`` mapped to a ``Activity``.

        arg:    activity_id (osid.id.Id): ``Id`` of a ``Activity``
        return: (osid.id.IdList) - list of objective bank ``Ids``
        raise:  NotFound - ``activity_id`` is not found
        raise:  NullArgument - ``activity_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_bin_ids_by_resource
        mgr = self._get_provider_manager('LEARNING', local=True)
        lookup_session = mgr.get_activity_lookup_session(proxy=self._proxy)
        lookup_session.use_federated_objective_bank_view()
        activity = lookup_session.get_activity(activity_id)
        id_list = []
        for idstr in activity._my_map['assignedObjectiveBankIds']:
            id_list.append(Id(idstr))
        return IdList(id_list)