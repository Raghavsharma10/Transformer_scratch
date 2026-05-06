def can_create_repository_with_record_types(self, repository_record_types=None):
        """Tests if this user can create a single ``Repository`` using the desired record types.

        While ``RepositoryManager.getRepositoryRecordTypes()`` can be
        used to examine which records are supported, this method tests
        which record(s) are required for creating a specific
        ``Repository``. Providing an empty array tests if a
        ``Repository`` can be created with no records.

        :param repository_record_types: array of repository record types
        :type repository_record_types: ``osid.type.Type[]``
        :return: ``true`` if ``Repository`` creation using the specified ``Types`` is supported, ``false`` otherwise
        :rtype: ``boolean``
        :raise: ``NullArgument`` -- ``repository_record_types`` is ``null``

        *compliance: mandatory -- This method must be implemented.*

        """
        url_path = construct_url('authorization',
                                 bank_id=self._catalog_idstr)
        return self._get_request(url_path)['objectiveBankHints']['canCreate']