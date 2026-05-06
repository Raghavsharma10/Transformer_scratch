def get_function(self):
        """Gets the ``Function`` for this authorization.

        return: (osid.authorization.Function) - the function
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.Activity.get_objective
        if not bool(self._my_map['functionId']):
            raise errors.IllegalState('function empty')
        mgr = self._get_provider_manager('AUTHORIZATION')
        if not mgr.supports_function_lookup():
            raise errors.OperationFailed('Authorization does not support Function lookup')
        lookup_session = mgr.get_function_lookup_session(proxy=getattr(self, "_proxy", None))
        lookup_session.use_federated_vault_view()
        return lookup_session.get_function(self.get_function_id())