def get_book_hierarchy_session(self, proxy):
        """Gets the ``OsidSession`` associated with the book hierarchy service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.commenting.BookHierarchySession) - a
                ``BookHierarchySession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_book_hierarchy()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_book_hierarchy()`` is ``true``.*

        """
        if not self.supports_book_hierarchy():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.BookHierarchySession(proxy=proxy, runtime=self._runtime)