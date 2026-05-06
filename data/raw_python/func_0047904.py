def create_asset_content(self, asset_content_form=None):
        """Creates new ``AssetContent`` for a given asset.

        arg:    asset_content_form (osid.repository.AssetContentForm):
                the form for this ``AssetContent``
        return: (osid.repository.AssetContent) - the new
                ``AssetContent``
        raise:  IllegalState - ``asset_content_form`` already used in a
                create transaction
        raise:  InvalidArgument - one or more of the form elements is
                invalid
        raise:  NullArgument - ``asset_content_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``asset_content_form`` did not originate
                from ``get_asset_content_form_for_create()``
        *compliance: mandatory -- This method must be implemented.*

        """
        if isinstance(asset_content_form, AssetContentForm):
            asset_content = self._provider_session.create_asset_content(
                asset_content_form._payload)
        else:
            asset_content = self._provider_session.create_asset_content(
                asset_content_form)
        try:
            if asset_content.has_url() and 'amazonaws.com' in asset_content.get_url():
                return AssetContent(asset_content, self._config_map)
        except TypeError:
            pass
        return asset_content