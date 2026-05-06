def get_comment_lookup_session_for_book(self, book_id, proxy):
        """Gets the ``OsidSession`` associated with the comment lookup service for the given book.

        arg:    book_id (osid.id.Id): the ``Id`` of the ``Book``
        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.commenting.CommentLookupSession) - a
                ``CommentLookupSession``
        raise:  NotFound - no ``Book`` found by the given ``Id``
        raise:  NullArgument - ``book_id`` or ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_comment_lookup()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_comment_lookup()`` and
        ``supports_visible_federation()`` are ``true``*

        """
        if not self.supports_comment_lookup():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.CommentLookupSession(book_id, proxy, self._runtime)