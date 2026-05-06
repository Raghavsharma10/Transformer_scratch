def find_gene(self, name: str):
        """Find gene(s) by name."""
        result = [ExpGene.from_series(s)
                  for i, s in self.loc[self['name'] == name].iterrows()]
        return result