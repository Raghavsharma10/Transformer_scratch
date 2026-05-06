def get_composition_admin_session(self):
        """Gets a composition administration session for creating, updating and deleting compositions.

        return: (osid.repository.CompositionAdminSession) - a
                ``CompositionAdminSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_composition_admin()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_composition_admin()`` is ``true``.*

        """
        if not self.supports_composition_admin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.CompositionAdminSession(runtime=self._runtime)