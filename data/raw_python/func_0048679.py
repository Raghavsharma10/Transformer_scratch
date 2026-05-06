def get_avatar(self):
        """Gets the asset.

        return: (osid.repository.Asset) - the asset
        raise:  IllegalState - ``has_avatar()`` is ``false``
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.Resource.get_avatar_template
        if not bool(self._my_map['avatarId']):
            raise errors.IllegalState('this Resource has no avatar')
        mgr = self._get_provider_manager('REPOSITORY')
        if not mgr.supports_asset_lookup():
            raise errors.OperationFailed('Repository does not support Asset lookup')
        lookup_session = mgr.get_asset_lookup_session(proxy=getattr(self, "_proxy", None))
        lookup_session.use_federated_repository_view()
        osid_object = lookup_session.get_asset(self.get_avatar_id())
        return osid_object