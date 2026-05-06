def update_asset(self, asset_form=None):
        """Updates an existing asset.

        arg:    asset_form (osid.repository.AssetForm): the form
                containing the elements to be updated
        raise:  IllegalState - ``asset_form`` already used in anupdate
                transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``asset_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``asset_form`` did not originate from
                ``get_asset_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        return Asset(self._provider_session.update_asset(asset_form), self._config_map)