def getCallSetId(self, sampleName):
        """
        Returns the callSetId for the specified sampleName in this
        VariantSet.
        """
        compoundId = datamodel.CallSetCompoundId(
            self.getCompoundId(), sampleName)
        return str(compoundId)