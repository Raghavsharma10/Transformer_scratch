def get_activity_admin_session(self, proxy):
        """Gets the ``OsidSession`` associated with the activity administration service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.learning.ActivityAdminSession) - an
                ``ActivityAdminSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_activity_admin()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_activity_admin()`` is ``true``.*

        """
        if not self.supports_activity_admin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ActivityAdminSession(proxy=proxy, runtime=self._runtime)