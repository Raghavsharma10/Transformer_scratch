def set_journal_comment(self, comment):
        """Sets a comment.

        arg:    comment (string): the new comment
        raise:  InvalidArgument - ``comment`` is invalid
        raise:  NoAccess - ``Metadata.isReadonly()`` is ``true``
        raise:  NullArgument - ``comment`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        self._my_map['journal_comment'] = self._get_display_text(comment, self.get_journal_comment_metadata())