def get_repositories_by_ids(self, repository_ids=None):
        """Gets a ``RepositoryList`` corresponding to the given ``IdList``.

        In plenary mode, the returned list contains all of the
        repositories specified in the ``Id`` list, in the order of the
        list, including duplicates, or an error results if an ``Id`` in
        the supplied list is not found or inaccessible. Otherwise,
        inaccessible ``Repositories`` may be omitted from the list and
        may present the elements in any order including returning a
        unique set.

        arg:    repository_ids (osid.id.IdList): the list of ``Ids`` to
                retrieve
        return: (osid.repository.RepositoryList) - the returned
                ``Repository list``
        raise:  NotFound - an ``Id`` was not found
        raise:  NullArgument - ``repository_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from awsosid template for -
        # osid.resource.BinLookupSession.get_bins_by_ids_template
        if not self._can('lookup'):
            raise PermissionDenied()
        else:
            return self._provider_session.get_repositories_by_ids(repository_ids)