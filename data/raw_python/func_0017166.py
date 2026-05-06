def edit(self, body):
        """Edit this comment.

        :param str body: (required), new body of the comment, Markdown
            formatted
        :returns: bool
        """
        if body:
            json = self._json(self._patch(self._api,
                              data=dumps({'body': body})), 200)
            if json:
                self._update_(json)
                return True
        return False