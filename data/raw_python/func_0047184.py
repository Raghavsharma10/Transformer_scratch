def get_objective_banks_by_activity(self, activity_id):
        """Gets the list of ``ObjectiveBanks`` mapped to a ``Activity``.

        arg:    activity_id (osid.id.Id): ``Id`` of a ``Activity``
        return: (osid.learning.ObjectiveBankList) - list of objective
                bank ``Ids``
        raise:  NotFound - ``activity_id`` is not found
        raise:  NullArgument - ``activity_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_bins_by_resource
        mgr = self._get_provider_manager('LEARNING', local=True)
        lookup_session = mgr.get_objective_bank_lookup_session(proxy=self._proxy)
        return lookup_session.get_objective_banks_by_ids(
            self.get_objective_bank_ids_by_activity(activity_id))