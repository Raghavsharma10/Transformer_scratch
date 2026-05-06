def get_asset_form_for_create(self, asset_record_types=None):
        """Gets the asset form for creating new assets.

        A new form should be requested for each create transaction.

        :param asset_record_types: array of asset record types
        :type asset_record_types: ``osid.type.Type[]``
        :return: the asset form
        :rtype: ``osid.repository.AssetForm``
        :raise: ``NullArgument`` -- ``asset_record_types`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``PermissionDenied`` -- authorization failure
        :raise: ``Unsupported`` -- unable to get form for requested record types

        *compliance: mandatory -- This method must be implemented.*

        """
        if asset_record_types is None:
            pass  # Still need to deal with the record_types argument
        asset_form = objects.AssetForm()
        self._forms[asset_form.get_id().get_identifier()] = not CREATED
        return asset_form