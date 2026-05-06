def get_book_lookup_session(self):
        """Gets the ``OsidSession`` associated with the book lookup service.

        return: (osid.commenting.BookLookupSession) - a
                ``BookLookupSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_book_lookup()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_book_lookup()`` is ``true``.*

        """
        if not self.supports_book_lookup():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.BookLookupSession(runtime=self._runtime)