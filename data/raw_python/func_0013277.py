def addCallSetFromName(self, sampleName):
        """
        Adds a CallSet for the specified sample name.
        """
        callSet = CallSet(self, sampleName)
        self.addCallSet(callSet)