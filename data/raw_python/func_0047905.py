def get_asset_content_form_for_update(self, asset_content_id=None):
        """Gets the asset content form for updating an existing asset content.

        A new asset content form should be requested for each update
        transaction.

        arg:    asset_content_id (osid.id.Id): the ``Id`` of the
                ``AssetContent``
        return: (osid.repository.AssetContentForm) - the asset content
                form
        raise:  NotFound - ``asset_content_id`` is not found
        raise:  NullArgument - ``asset_content_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        asset_content_form = self._provider_session.get_asset_content_form_for_update(
            asset_content_id)
        if 'amazonaws.com' in asset_content_form.get_url_metadata().get_existing_string_values()[0]:
            return AssetContentForm(asset_content_form,
                                    self._config_map,
                                    self.get_repository_id())
        return asset_content_form