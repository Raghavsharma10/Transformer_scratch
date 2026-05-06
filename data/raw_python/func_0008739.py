def as_list(self):
        """
        Return an *ordered* list of the source attributes
        """
        self._sanitise()
        l = []
        for name in self.names:
            l.append(getattr(self, name))
        return l