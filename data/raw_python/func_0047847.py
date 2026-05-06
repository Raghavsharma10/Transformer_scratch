def get_asset_form_for_update(self, asset_id=None):
        """Gets the asset form for updating an existing asset.

        A new asset form should be requested for each update
        transaction.

        :param asset_id: the ``Id`` of the ``Asset``
        :type asset_id: ``osid.id.Id``
        :return: the asset form
        :rtype: ``osid.repository.AssetForm``
        :raise: ``NotFound`` -- ``asset_id`` is not found
        :raise: ``NullArgument`` -- ``asset_id`` is null
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``PermissionDenied`` -- authorization failure

        *compliance: mandatory -- This method must be implemented.*

        """
        if asset_id is None:
            raise NullArgument()
        try:
            url_path = construct_url('assets',
                                     bank_id=self._catalog_idstr,
                                     asset_id=asset_id)
            asset = objects.Asset(self._get_request(url_path))
        except Exception:
            raise
        asset_form = objects.AssetForm(asset._my_map)
        self._forms[asset_form.get_id().get_identifier()] = not UPDATED
        return asset_form