def reMutualReceptions(self, idA, idB):
        """ Returns ruler and exaltation mutual receptions. """
        mr = self.mutualReceptions(idA, idB)
        filter_ = ['ruler', 'exalt']
        # Each pair of dignities must be 'ruler' or 'exalt'
        return [(a,b) for (a,b) in mr if (a in filter_ and b in filter_)]