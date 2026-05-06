def get_qualifier(self):
        """Gets the qualifier for this authorization.

        return: (osid.authorization.Qualifier) - the qualifier
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.Activity.get_objective
        if not bool(self._my_map['qualifierId']):
            raise errors.IllegalState('qualifier empty')
        mgr = self._get_provider_manager('AUTHORIZATION')
        if not mgr.supports_qualifier_lookup():
            raise errors.OperationFailed('Authorization does not support Qualifier lookup')
        lookup_session = mgr.get_qualifier_lookup_session(proxy=getattr(self, "_proxy", None))
        lookup_session.use_federated_vault_view()
        return lookup_session.get_qualifier(self.get_qualifier_id())