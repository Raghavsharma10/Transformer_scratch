def channel(self, match):
        """ Return Channel object for a given Slack ID or name """
        if len(match) == 9 and match[0] in ('C','G','D'):
            return self._lookup(Channel, 'id', match)
        return self._lookup(Channel, 'name', match)