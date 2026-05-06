def can_delete_asset_contents(self, asset_id=None):
        """Tests if this user can delete ``AssetsContent`` from ``Assets``.

        A return of true does not guarantee successful authorization. A
        return of false indicates that it is known deleting an
        ``AssetContent`` will result in a ``PermissionDenied``. This is
        intended as a hint to an application that may opt not to offer
        delete operations to an unauthorized user.

        :param asset_id: the ``Id`` of an ``Asset``
        :type asset_id: ``osid.id.Id``
        :return: ``false`` if ``AssetContent`` deletion is not authorized, ``true`` otherwise
        :rtype: ``boolean``
        :raise: ``NullArgument`` -- ``asset_id`` is ``null``

        *compliance: mandatory -- This method must be implemented.*

        """
        url_path = construct_url('authorization',
                                 bank_id=self._catalog_idstr)
        return self._get_request(url_path)['assetHints']['canDelete']