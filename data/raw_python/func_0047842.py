def get_assets_by_genus_type(self, asset_genus_type=None):
        """Gets an ``AssetList`` corresponding to the given asset genus ``Type`` which does not
        include assets of types derived from the specified ``Type``.

        In plenary mode, the returned list contains all known assets or
        an error results. Otherwise, the returned list may contain only
        those assets that are accessible through this session.

        :param asset_genus_type: an asset genus type
        :type asset_genus_type: ``osid.type.Type``
        :return: the returned ``Asset list``
        :rtype: ``osid.repository.AssetList``
        :raise: ``NullArgument`` -- ``asset_genus_type`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``PermissionDenied`` -- authorization failure

        *compliance: mandatory -- This method must be implemented.*

        """
        if asset_genus_type is None:
            raise NullArgument()
        url_path = construct_url('assets_by_genus',
                                 bank_id=self._catalog_idstr,
                                 genus_type=asset_genus_type.get_identifier())
        return objects.AssetList(self._get_request(url_path))