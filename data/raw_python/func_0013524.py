def getBiosampleByName(self, name):
        """
        Returns a Biosample with the specified name, or raises a
        BiosampleNameNotFoundException if it does not exist.
        """
        if name not in self._biosampleNameMap:
            raise exceptions.BiosampleNameNotFoundException(name)
        return self._biosampleNameMap[name]