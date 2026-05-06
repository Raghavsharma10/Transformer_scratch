def get_asset_content_form_for_create(self, asset_id=None, asset_content_record_types=None):
        """Gets an asset content form for creating new assets.

        :param asset_id: the ``Id`` of an ``Asset``
        :type asset_id: ``osid.id.Id``
        :param asset_content_record_types: array of asset content record types
        :type asset_content_record_types: ``osid.type.Type[]``
        :return: the asset content form
        :rtype: ``osid.repository.AssetContentForm``
        :raise: ``NotFound`` -- ``asset_id`` is not found
        :raise: ``NullArgument`` -- ``asset_id`` or ``asset_content_record_types`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``PermissionDenied`` -- authorization failure
        :raise: ``Unsupported`` -- unable to get form for requested record types

        *compliance: mandatory -- This method must be implemented.*

        """
        if asset_id is None:
            raise NullArgument()
        if asset_content_record_types is None:
            pass  # Still need to deal with the record_types argument
        asset_content_form = objects.AssetContentForm(asset_id=asset_id)
        self._forms[asset_content_form.get_id().get_identifier()] = not CREATED
        return asset_content_form