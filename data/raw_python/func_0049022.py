def get_comment_admin_session(self, proxy):
        """Gets the ``OsidSession`` associated with the comment administration service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.commenting.CommentAdminSession) - a
                ``CommentAdminSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_comment_admin()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_comment_admin()`` is ``true``.*

        """
        if not self.supports_comment_admin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.CommentAdminSession(proxy=proxy, runtime=self._runtime)