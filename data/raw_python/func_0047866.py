def get_repository_form_for_create(self, repository_record_types=None):
        """Gets the repository form for creating new repositories.

        A new form should be requested for each create transaction.

        :param repository_record_types: array of repository record types
        :type repository_record_types: ``osid.type.Type[]``
        :return: the repository form
        :rtype: ``osid.repository.RepositoryForm``
        :raise: ``NullArgument`` -- ``repository_record_types`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``PermissionDenied`` -- authorization failure
        :raise: ``Unsupported`` -- unable to get form for requested record types

        *compliance: mandatory -- This method must be implemented.*

        """
        if repository_record_types is None:
            pass  # Still need to deal with the record_types argument
        repository_form = objects.RepositoryForm()
        self._forms[repository_form.get_id().get_identifier()] = not CREATED
        return repository_form