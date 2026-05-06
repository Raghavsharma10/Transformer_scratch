def queueTypeUpgrade(self, oldtype):
        """
        Queue a type upgrade for C{oldtype}.
        """
        if oldtype not in self._oldTypesRemaining:
            self._oldTypesRemaining.append(oldtype)