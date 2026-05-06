def _handle_comment(self, comment):
        """Deal with a comment."""
        if not comment:
            return ''
        start = self.indent_type
        if not comment.startswith('#'):
            start += self._a_to_u(' # ')
        return (start + comment)