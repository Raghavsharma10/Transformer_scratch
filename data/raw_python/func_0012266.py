def comment(self, text, comment_prefix='#'):
        """Creates a comment block

        Args:
            text (str): content of comment without #
            comment_prefix (str): character indicating start of comment

        Returns:
            self for chaining
        """
        comment = Comment(self._container)
        if not text.startswith(comment_prefix):
            text = "{} {}".format(comment_prefix, text)
        if not text.endswith('\n'):
            text = "{}{}".format(text, '\n')
        comment.add_line(text)
        self._container.structure.insert(self._idx, comment)
        self._idx += 1
        return self