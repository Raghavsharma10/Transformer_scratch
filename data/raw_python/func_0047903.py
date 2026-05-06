def get_asset_content_form_for_create(self,
                                          asset_id=None,
                                          asset_content_record_types=None):
        """Gets an asset content form for creating new assets.

        arg:    asset_id (osid.id.Id): the ``Id`` of an ``Asset``
        arg:    asset_content_record_types (osid.type.Type[]): array of
                asset content record types
        return: (osid.repository.AssetContentForm) - the asset content
                form
        raise:  NotFound - ``asset_id`` is not found
        raise:  NullArgument - ``asset_id`` or
                ``asset_content_record_types`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - unable to get form for requested record
                types
        *compliance: mandatory -- This method must be implemented.*

        """
        if AWS_ASSET_CONTENT_RECORD_TYPE in asset_content_record_types:
            asset_content_record_types.remove(AWS_ASSET_CONTENT_RECORD_TYPE)
            return AssetContentForm(
                self._provider_session.get_asset_content_form_for_create(
                    asset_id,
                    asset_content_record_types),
                self._config_map,
                self.get_repository_id())
        else:
            return self._provider_session.get_asset_content_form_for_create(
                asset_id,
                asset_content_record_types)