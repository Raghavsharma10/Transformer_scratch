def mappings(self):
        """
        :class:`caspo.core.mapping.MappingList`: the list of mappings present in at least one logical network in this list
        """
        return self.hg.mappings[np.unique(np.where(self.__matrix == 1)[1])]