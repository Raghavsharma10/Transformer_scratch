def get_asset_contents(self):
        """Gets the content of this asset.

        return: (osid.repository.AssetContentList) - the asset contents
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.repository.Asset.get_asset_contents_template
        return AssetContentList(
            self._my_map['assetContents'],
            runtime=self._runtime,
            proxy=self._proxy)