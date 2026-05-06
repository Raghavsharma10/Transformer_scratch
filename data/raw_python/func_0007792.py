def inDignities(self, idA, idB):
        """ Returns the dignities of A which belong to B. """
        objA = self.chart.get(idA)
        info = essential.getInfo(objA.sign, objA.signlon)
        # Should we ignore exile and fall?
        return [dign for (dign, ID) in info.items() if ID == idB]