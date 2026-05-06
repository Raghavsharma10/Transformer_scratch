def get_asset(self):
        """Gets the ``Asset`` corresponding to this content.

        return: (osid.repository.Asset) - the asset
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.Activity.get_objective
        if not bool(self._my_map['assetId']):
            raise errors.IllegalState('asset empty')
        mgr = self._get_provider_manager('REPOSITORY')
        if not mgr.supports_asset_lookup():
            raise errors.OperationFailed('Repository does not support Asset lookup')
        lookup_session = mgr.get_asset_lookup_session(proxy=getattr(self, "_proxy", None))
        lookup_session.use_federated_repository_view()
        return lookup_session.get_asset(self.get_asset_id())