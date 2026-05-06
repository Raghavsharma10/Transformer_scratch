def get_repository_form_for_create(self, repository_record_types=None):
        """Gets the repository form for creating new repositories.

        A new form should be requested for each create transaction.

        arg:    repository_record_types (osid.type.Type[]): array of
                repository record types
        return: (osid.repository.RepositoryForm) - the repository form
        raise:  NullArgument - ``repository_record_types`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - unable to get form for requested record
                types
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from awsosid template for -
        # osid.resource.BinAdminSession.get_bin_form_for_create_template
        if not self._can('create'):
            raise PermissionDenied()
        else:
            return self._provider_session.get_repository_form_for_create(repository_record_types)