def getBiosample(self, id_):
        """
        Returns the Biosample with the specified id, or raises
        a BiosampleNotFoundException otherwise.
        """
        if id_ not in self._biosampleIdMap:
            raise exceptions.BiosampleNotFoundException(id_)
        return self._biosampleIdMap[id_]