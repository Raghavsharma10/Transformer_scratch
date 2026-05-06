def delete_asset(self, asset_id=None):
        """Deletes an ``Asset``.

        :param asset_id: the ``Id`` of the ``Asset`` to remove
        :type asset_id: ``osid.id.Id``
        :raise: ``NotFound`` -- ``asset_id`` not found
        :raise: ``NullArgument`` -- ``asset_id`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``PermissionDenied`` -- authorization failure

        *compliance: mandatory -- This method must be implemented.*

        """
        if asset_id is None:
            raise NullArgument()
        if not isinstance(asset_id, Id):
            raise InvalidArgument('argument type is not an osid Id')

        url_path = construct_url('assets',
                                 bank_id=self._catalog_idstr,
                                 asset_id=asset_id)
        result = self._delete_request(url_path)
        return objects.Asset(result)