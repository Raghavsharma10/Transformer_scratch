def get_repositories_by_ids(self, repository_ids=None):
        """Gets a ``RepositoryList`` corresponding to the given ``IdList``.

        In plenary mode, the returned list contains all of the
        repositories specified in the ``Id`` list, in the order of the
        list, including duplicates, or an error results if an ``Id`` in
        the supplied list is not found or inaccessible. Otherwise,
        inaccessible ``Repositories`` may be omitted from the list and
        may present the elements in any order including returning a
        unique set.

        :param repository_ids: the list of ``Ids`` to retrieve
        :type repository_ids: ``osid.id.IdList``
        :return: the returned ``Repository list``
        :rtype: ``osid.repository.RepositoryList``
        :raise: ``NotFound`` -- an ``Id`` was not found
        :raise: ``NullArgument`` -- ``repository_ids`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``PermissionDenied`` -- authorization failure

        *compliance: mandatory -- This method must be implemented.*

        """
        if repository_ids is None:
            raise NullArgument()
        repositories = []
        for i in repository_ids:
            repository = None
            url_path = construct_url('objective_banks',
                                     bank_id=i)
            try:
                repository = self._get_request(url_path)
            except (NotFound, OperationFailed):
                if self._repository_view == PLENARY:
                    raise
                else:
                    pass
            if repository:
                if not (self._repository_view == COMPARATIVE and
                        repository in repositories):
                    repositories.append(repository)
        return objects.RepositoryList(repositories)