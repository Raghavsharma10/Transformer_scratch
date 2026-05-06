def can_delete_repositories(self):
        """Tests if this user can delete ``Repositories``.

        A return of true does not guarantee successful authorization. A
        return of false indicates that it is known deleting a
        ``Repository`` will result in a ``PermissionDenied``. This is
        intended as a hint to an application that may not wish to offer
        delete operations to unauthorized users.

        :return: ``false`` if ``Repository`` deletion is not authorized, ``true`` otherwise
        :rtype: ``boolean``


        *compliance: mandatory -- This method must be implemented.*

        """
        url_path = construct_url('authorization',
                                 bank_id=self._catalog_idstr)
        return self._get_request(url_path)['objectiveBankHints']['canDelete']