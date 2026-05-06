def get_comment_lookup_session(self):
        """Gets the ``OsidSession`` associated with the comment lookup service.

        return: (osid.commenting.CommentLookupSession) - a
                ``CommentLookupSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_comment_lookup()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_comment_lookup()`` is ``true``.*

        """
        if not self.supports_comment_lookup():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.CommentLookupSession(runtime=self._runtime)