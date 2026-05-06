def genes(self):
        """Return a list of all genes."""
        return [ExpGene.from_series(g)
                for i, g in self.reset_index().iterrows()]