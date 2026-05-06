def get_repositories_by_genus_type(self, repository_genus_type=None):
        """Gets a ``RepositoryList`` corresponding to the given repository genus ``Type`` which
            does not include repositories of types derived from the specified ``Type``.

        In plenary mode, the returned list contains all known
        repositories or an error results. Otherwise, the returned list
        may contain only those repositories that are accessible through
        this session.

        :param repository_genus_type: a repository genus type
        :type repository_genus_type: ``osid.type.Type``
        :return: the returned ``Repository list``
        :rtype: ``osid.repository.RepositoryList``
        :raise: ``NullArgument`` -- ``repository_genus_type`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``PermissionDenied`` -- authorization failure

        *compliance: mandatory -- This method must be implemented.*

        """
        if repository_genus_type is None:
            raise NullArgument()
        url_path = construct_url('objective_banks')
        repositories_of_type = []
        all_repositories = self._get_request(url_path)
        for repository in all_repositories:
            # DO WE NEED TO CHECK ALL THREE ATRIBUTES OF THE Id HERE?
            if repository['genusTypeId'] == repository_genus_type.get_identifier():
                repositories_of_type.append[repository]
        return objects.RepositoryList(repositories_of_type)