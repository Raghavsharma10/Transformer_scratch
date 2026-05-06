def mutualReceptions(self, idA, idB):
        """ Returns all pairs of dignities in mutual reception. """
        AB = self.receives(idA, idB)
        BA = self.receives(idB, idA)
        # Returns a product of both lists
        return [(a,b) for a in AB for b in BA]