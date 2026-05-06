def getLeaves(self):
        """Return the downstream leaf stages of this stage."""
        result = list()
        if not self._next_stages:
            result.append(self)
        else:
            for stage in self._next_stages:
                leaves = stage.getLeaves()
                result += leaves
        return result