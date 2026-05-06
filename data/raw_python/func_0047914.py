def delete_repository(self, repository_id=None):
        """Deletes a ``Repository``.

        arg:    repository_id (osid.id.Id): the ``Id`` of the
                ``Repository`` to remove
        raise:  NotFound - ``repository_id`` not found
        raise:  NullArgument - ``repository_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from awsosid template for -
        # osid.resource.BinAdminSession.delete_bin_template
        if not self._can('delete'):
            raise PermissionDenied()
        else:
            return self._provider_session.delete_repository(repository_id)