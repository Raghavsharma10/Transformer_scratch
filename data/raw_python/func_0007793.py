def receives(self, idA, idB):
        """ Returns the dignities where A receives B.
        A receives B when (1) B aspects A and (2) B is in 
        dignities of A.

        """
        objA = self.chart.get(idA)
        objB = self.chart.get(idB)
        asp = aspects.isAspecting(objB, objA, const.MAJOR_ASPECTS)
        return self.inDignities(idB, idA) if asp else []