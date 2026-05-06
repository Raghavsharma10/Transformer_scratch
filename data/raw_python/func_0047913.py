def update_repository(self, repository_form=None):
        """Updates an existing repository.

        arg:    repository_form (osid.repository.RepositoryForm): the
                form containing the elements to be updated
        raise:  IllegalState - ``repository_form`` already used in an
                update transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``repository_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``repository_form`` did not originate from
                ``get_repository_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from awsosid template for -
        # osid.resource.BinAdminSession.update_bin_template
        if not self._can('update'):
            raise PermissionDenied()
        else:
            return self._provider_session.update_repository(repository_form)