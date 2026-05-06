def get_comment_query_session(self, proxy):
        """Gets the ``OsidSession`` associated with the comment query service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.commenting.CommentQuerySession) - a
                ``CommentQuerySession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_comment_query()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_comment_query()`` is ``true``.*

        """
        if not self.supports_comment_query():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.CommentQuerySession(proxy=proxy, runtime=self._runtime)