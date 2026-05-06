def can_update_assets(self):
        """Tests if this user can update ``Assets``.

        A return of true does not guarantee successful authorization. A
        return of false indicates that it is known updating an ``Asset``
        will result in a ``PermissionDenied``. This is intended as a
        hint to an application that may opt not to offer update
        operations to an unauthorized user.

        :return: ``false`` if ``Asset`` modification is not authorized, ``true`` otherwise
        :rtype: ``boolean``


        *compliance: mandatory -- This method must be implemented.*

        """
        url_path = construct_url('authorization',
                                 bank_id=self._catalog_idstr)
        return self._get_request(url_path)['assetHints']['canUpdate']