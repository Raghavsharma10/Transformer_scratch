def get_comment_book_session(self):
        """Gets the session for retrieving comment to book mappings.

        return: (osid.commenting.CommentBookSession) - a
                ``CommentBookSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_comment_book()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_comment_book()`` is ``true``.*

        """
        if not self.supports_comment_book():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.CommentBookSession(runtime=self._runtime)