def cardinal_groupby(self):
        """
        Group this object on it cardinal dimension (_cardinal).

        Returns:
            grpby: Pandas groupby object (grouped on _cardinal)
        """
        g, t = self._cardinal
        self[g] = self[g].astype(t)
        grpby = self.groupby(g)
        self[g] = self[g].astype('category')
        return grpby