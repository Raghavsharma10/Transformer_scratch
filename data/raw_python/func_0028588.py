def user(self, match):
        """ Return User object for a given Slack ID or name """
        if len(match) == 9 and match[0] == 'U':
            return self._lookup(User, 'id', match)
        return self._lookup(User, 'name', match)