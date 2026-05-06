def request_name(self, name):
        """Request a name, might return the name or a similar one if already
        used or reserved
        """

        while name in self._blacklist:
            name += "_"
        self._blacklist.add(name)
        return name