def lookupFunction(self, proto, name, namespace):
        """Return a callable to invoke when executing the named command.
        """
        # Try to find a method to be invoked in a transaction first
        # Otherwise fallback to a "regular" method
        fName = self.autoDispatchPrefix + name
        fObj = getattr(self, fName, None)
        if fObj is not None:
            # pass the namespace along
            return self._auto(fObj, proto, namespace)

        assert namespace is None, 'Old-style parsing'
        # Fall back to simplistic command dispatching - we probably want to get
        # rid of this eventually, there's no reason to do extra work and write
        # fewer docs all the time.
        fName = self.baseDispatchPrefix + name
        return getattr(self, fName, None)