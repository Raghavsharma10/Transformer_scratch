def delete_asset_content(self, asset_content_id=None):
        """Deletes content from an ``Asset``.

        arg:    asset_content_id (osid.id.Id): the ``Id`` of the
                ``AssetContent``
        raise:  NotFound - ``asset_content_id`` is not found
        raise:  NullArgument - ``asset_content_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        asset_content = self._get_asset_content(asset_content_id)
        if asset_content.has_url() and 'amazonaws.com' in asset_content.get_url():
            # print "Still have to implement removing files from aws"
            key = asset_content.get_url().split('amazonaws.com')[1]
            remove_file(self._config_map, key)
            self._provider_session.delete_asset_content(asset_content_id)
        else:
            self._provider_session.delete_asset_content(asset_content_id)