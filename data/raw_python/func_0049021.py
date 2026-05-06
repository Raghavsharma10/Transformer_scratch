def get_comment_query_session_for_book(self, book_id, proxy):
        """Gets the ``OsidSession`` associated with the comment query service for the given book.

        arg:    book_id (osid.id.Id): the ``Id`` of the ``Book``
        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.commenting.CommentQuerySession) - a
                ``CommentQuerySession``
        raise:  NotFound - no ``Comment`` found by the given ``Id``
        raise:  NullArgument - ``book_id`` or ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_comment_query()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_comment_query()`` and
        ``supports_visible_federation()`` are ``true``*

        """
        if not self.supports_comment_query():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.CommentQuerySession(book_id, proxy, self._runtime)