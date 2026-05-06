def get_asset_contents_for_asset(self, asset_id):
        """Gets an ``AssetList`` from the given Asset.

        In plenary mode, the returned list contains all known asset contents or
        an error results. Otherwise, the returned list may contain only
        those asset contents that are accessible through this session.

        :param asset_id: an asset ``Id``
        :type asset_id: ``osid.id.Id``
        :return: the returned ``AssetContent list``
        :rtype: ``osid.repository.AssetContentList``
        :raise: ``NullArgument`` -- ``asset_id`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``PermissionDenied`` -- authorization failure

        *compliance: mandatory -- This method must be implemented.*

        """
        return AssetContentList(self._provider_session.get_asset_contents_for_asset(asset_id), self._config_map)