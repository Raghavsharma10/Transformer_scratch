def get_proficiency_admin_session(self, proxy):
        """Gets the ``OsidSession`` associated with the proficiency administration service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.learning.ProficiencyAdminSession) - a
                ``ProficiencyAdminSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_proficiency_admin()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_proficiency_admin()`` is ``true``.*

        """
        if not self.supports_proficiency_admin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ProficiencyAdminSession(proxy=proxy, runtime=self._runtime)