def get_gradebook_admin_session(self, proxy):
        """Gets the OsidSession associated with the gradebook administration service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.grading.GradebookAdminSession) - a
                ``GradebookAdminSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_gradebook_admin() is false``
        *compliance: optional -- This method must be implemented if
        ``supports_gradebook_admin()`` is true.*

        """
        if not self.supports_gradebook_admin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.GradebookAdminSession(proxy=proxy, runtime=self._runtime)