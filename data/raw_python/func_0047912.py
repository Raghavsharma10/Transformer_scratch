def get_repository_form_for_update(self, repository_id=None):
        """Gets the repository form for updating an existing repository.

        A new repository form should be requested for each update
        transaction.

        arg:    repository_id (osid.id.Id): the ``Id`` of the
                ``Repository``
        return: (osid.repository.RepositoryForm) - the repository form
        raise:  NotFound - ``repository_id`` is not found
        raise:  NullArgument - ``repository_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from awsosid template for -
        # osid.resource.BinAdminSession.get_bin_form_for_update_template
        if not self._can('update'):
            raise PermissionDenied()
        else:
            return self._provider_session.get_repository_form_for_update(repository_id)