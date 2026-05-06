def get_trust(self):
        """Gets the ``Trust`` for this authorization.

        return: (osid.authentication.process.Trust) - the ``Trust``
        raise:  IllegalState - ``has_trust()`` is ``false``
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.Resource.get_avatar_template
        if not bool(self._my_map['trustId']):
            raise errors.IllegalState('this Authorization has no trust')
        mgr = self._get_provider_manager('AUTHENTICATION.PROCESS')
        if not mgr.supports_trust_lookup():
            raise errors.OperationFailed('Authentication.Process does not support Trust lookup')
        lookup_session = mgr.get_trust_lookup_session(proxy=getattr(self, "_proxy", None))
        lookup_session.use_federated_agency_view()
        osid_object = lookup_session.get_trust(self.get_trust_id())
        return osid_object