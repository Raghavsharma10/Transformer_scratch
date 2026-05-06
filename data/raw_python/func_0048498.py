def get_resource(self):
        """Gets the ``Resource`` for this authorization.

        return: (osid.resource.Resource) - the ``Resource``
        raise:  IllegalState - ``has_resource()`` is ``false``
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.Resource.get_avatar_template
        if not bool(self._my_map['resourceId']):
            raise errors.IllegalState('this Authorization has no resource')
        mgr = self._get_provider_manager('RESOURCE')
        if not mgr.supports_resource_lookup():
            raise errors.OperationFailed('Resource does not support Resource lookup')
        lookup_session = mgr.get_resource_lookup_session(proxy=getattr(self, "_proxy", None))
        lookup_session.use_federated_bin_view()
        osid_object = lookup_session.get_resource(self.get_resource_id())
        return osid_object