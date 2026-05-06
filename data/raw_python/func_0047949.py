def set_journal_comment(self, comment=None):
        """Sets a comment.

        arg:    comment (string): the new comment
        raise:  InvalidArgument - comment is invalid
        raise:  NoAccess - metadata.is_readonly() is true
        raise:  NullArgument - comment is null
        compliance: mandatory - This method must be implemented.

        """
        if comment is None:
            raise NullArgument()
        metadata = Metadata(**settings.METADATA['comment'])
        if metadata.is_read_only():
            raise NoAccess()
        if self._is_valid_input(comment, metadata, array=False):
            self._my_map['journalComment']['text'] = comment
        else:
            raise InvalidArgument()