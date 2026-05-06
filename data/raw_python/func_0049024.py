def get_book_admin_session(self, proxy):
        """Gets the ``OsidSession`` associated with the book administrative service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.commenting.BookAdminSession) - a
                ``BookAdminSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_book_admin()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_book_admin()`` is ``true``.*

        """
        if not self.supports_book_admin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.BookAdminSession(proxy=proxy, runtime=self._runtime)