def get_comment_book_assignment_session(self):
        """Gets the session for assigning comment to book mappings.

        return: (osid.commenting.CommentBookAssignmentSession) - a
                ``CommentBookAssignmentSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_comment_book_assignment()``
                is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_comment_book_assignment()`` is ``true``.*

        """
        if not self.supports_comment_book_assignment():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.CommentBookAssignmentSession(runtime=self._runtime)