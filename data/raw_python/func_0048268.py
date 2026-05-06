def get_bin_admin_session(self):
        """Gets the bin administrative session for creating, updating and deleteing bins.

        return: (osid.resource.BinAdminSession) - a ``BinAdminSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_bin_admin()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_bin_admin()`` is ``true``.*

        """
        if not self.supports_bin_admin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.BinAdminSession(runtime=self._runtime)