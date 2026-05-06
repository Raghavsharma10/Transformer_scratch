def get_objectives_by_objective_bank(self, objective_bank_id):
        """Gets the list of ``Objectives`` associated with an ``ObjectiveBank``.

        arg:    objective_bank_id (osid.id.Id): ``Id`` of the
                ``ObjectiveBank``
        return: (osid.learning.ObjectiveList) - list of related
                objective ``Ids``
        raise:  NotFound - ``objective_bank_id`` is not found
        raise:  NullArgument - ``objective_bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resources_by_bin
        mgr = self._get_provider_manager('LEARNING', local=True)
        lookup_session = mgr.get_objective_lookup_session_for_objective_bank(objective_bank_id, proxy=self._proxy)
        lookup_session.use_isolated_objective_bank_view()
        return lookup_session.get_objectives()