def can_create_asset_content(self, asset_id=None):
        """Tests if this user can create content for ``Assets``.

        A return of true does not guarantee successful authorization. A
        return of false indicates that it is known creating an ``Asset``
        will result in a ``PermissionDenied``. This is intended as a
        hint to an application that may opt not to offer create
        operations to an unauthorized user.

        :param asset_id: the ``Id`` of an ``Asset``
        :type asset_id: ``osid.id.Id``
        :return: ``false`` if ``Asset`` content ceration is not authorized, ``true`` otherwise
        :rtype: ``boolean``
        :raise: ``NullArgument`` -- ``asset_id`` is ``null``

        *compliance: mandatory -- This method must be implemented.*

        """
        url_path = construct_url('authorization',
                                 bank_id=self._catalog_idstr)
        return self._get_request(url_path)['assetHints']['canCreate']