def get_asset_contents_by_query(self, asset_content_query=None):
        """Gets a list of ``AssetContents`` matching the given asset content query.

        arg:    asset_content_query (osid.repository.AssetContentQuery): the asset
                content query
        return: (osid.repository.AssetContentList) - the returned ``AssetContentList``
        raise:  NullArgument - ``asset_content_query`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - the ``asset_content_query`` is not of this service
        *compliance: mandatory -- This method must be implemented.*

        """
        return AssetContentList(self._provider_session.get_asset_contents_by_query(asset_content_query),
                                self._config_map)