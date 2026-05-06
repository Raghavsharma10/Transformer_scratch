def get_asset_contents_by_genus_type(self, asset_content_genus_type):
        """Gets an ``AssetContentList`` corresponding to the given asset content genus ``Type`` which does not include asset contents of types derived from the specified ``Type``.

        In plenary mode, the returned list contains all known asset contents or
        an error results. Otherwise, the returned list may contain only
        those asset contents that are accessible through this session.

        :param asset_content_genus_type: an asset content genus type
        :type asset_content_genus_type: ``osid.type.Type``
        :return: the returned ``AssetContent list``
        :rtype: ``osid.repository.AssetContentList``
        :raise: ``NullArgument`` -- ``asset_content_genus_type`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``PermissionDenied`` -- authorization failure

        *compliance: mandatory -- This method must be implemented.*

        """
        return AssetContentList(self._provider_session.get_asset_contents_by_genus_type(asset_content_genus_type), self._config_map)