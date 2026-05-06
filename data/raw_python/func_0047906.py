def update_asset_content(self, asset_content_form=None):
        """Updates an existing asset content.

        arg:    asset_content_form (osid.repository.AssetContentForm):
                the form containing the elements to be updated
        raise:  IllegalState - ``asset_content_form`` already used in an
                update transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``asset_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``asset_content_form`` did not originate
                from ``get_asset_content_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        if isinstance(asset_content_form, AssetContentForm):
            asset_content = self._provider_session.update_asset_content(
                asset_content_form._payload)
        else:
            asset_content = self._provider_session.update_asset_content(
                asset_content_form)
        if asset_content is not None and asset_content.has_url() and \
                'amazonaws.com' in asset_content.get_url():
            return AssetContent(asset_content, self._config_map)
        return asset_content