def getReadAlignmentId(self, gaAlignment):
        """
        Returns a string ID suitable for use in the specified GA
        ReadAlignment object in this ReadGroupSet.
        """
        compoundId = datamodel.ReadAlignmentCompoundId(
            self.getCompoundId(), gaAlignment.fragment_name)
        return str(compoundId)