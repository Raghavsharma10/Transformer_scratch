def lastLogged(self):
        """Get a dictionary of last logged messages.
        Keys are log types and values are the the last messages."""
        d = copy.deepcopy(self.__lastLogged)
        d.pop(-1, None)
        return d