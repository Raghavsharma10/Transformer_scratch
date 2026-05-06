def add_dependency(self, name, obj):
        """Add a code dependency so it gets inserted into globals"""

        if name in self._deps:
            if self._deps[name] is obj:
                return
            raise ValueError(
                "There exists a different dep with the same name : %r" % name)
        self._deps[name] = obj