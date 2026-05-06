def get_assets(self):
        """Gets any assets associated with this activity.

        return: (osid.repository.AssetList) - list of assets
        raise:  IllegalState - ``is_asset_based_activity()`` is
                ``false``
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.Activity.get_assets_template
        if not bool(self._my_map['assetIds']):
            raise errors.IllegalState('no assetIds')
        mgr = self._get_provider_manager('REPOSITORY')
        if not mgr.supports_asset_lookup():
            raise errors.OperationFailed('Repository does not support Asset lookup')

        # What about the Proxy?
        lookup_session = mgr.get_asset_lookup_session(proxy=getattr(self, "_proxy", None))
        lookup_session.use_federated_repository_view()
        return lookup_session.get_assets_by_ids(self.get_asset_ids())