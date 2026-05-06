def get_objective(self):
        """Gets the related objective.

        return: (osid.learning.Objective) - the related objective
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.Activity.get_objective
        if not bool(self._my_map['objectiveId']):
            raise errors.IllegalState('objective empty')
        mgr = self._get_provider_manager('LEARNING')
        if not mgr.supports_objective_lookup():
            raise errors.OperationFailed('Learning does not support Objective lookup')
        lookup_session = mgr.get_objective_lookup_session(proxy=getattr(self, "_proxy", None))
        lookup_session.use_federated_objective_bank_view()
        return lookup_session.get_objective(self.get_objective_id())