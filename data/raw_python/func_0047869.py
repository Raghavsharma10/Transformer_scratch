def get_repository_form_for_update(self, repository_id=None):
        """Gets the repository form for updating an existing repository.

        A new repository form should be requested for each update
        transaction.

        :param repository_id: the ``Id`` of the ``Repository``
        :type repository_id: ``osid.id.Id``
        :return: the repository form
        :rtype: ``osid.repository.RepositoryForm``
        :raise: ``NotFound`` -- ``repository_id`` is not found
        :raise: ``NullArgument`` -- ``repository_id`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``PermissionDenied`` -- authorization failure

        *compliance: mandatory -- This method must be implemented.*

        """
        if repository_id is None:
            raise NullArgument()
        try:
            repository = RepositoryLookupSession(proxy=self._proxy,
                                                 runtime=self._runtime).get_repository(repository_id)
        except Exception:
            raise
        repository_form = objects.RepositoryForm(repository._my_map)
        self._forms[repository_form.get_id().get_identifier()] = not UPDATED
        return repository_form