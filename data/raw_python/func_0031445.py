def newFromSites(self, sites, exclude=False):
        """
        Create a new read from self, with only certain sites.

        @param sites: A set of C{int} 0-based sites (i.e., indices) in
            sequences that should be kept. If C{None} (the default), all sites
            are kept.
        @param exclude: If C{True} the C{sites} will be excluded, not
            included.
        """
        if exclude:
            sites = set(range(len(self))) - sites

        newSequence = []
        newStructure = []
        for index, (base, structure) in enumerate(zip(self.sequence,
                                                      self.structure)):
            if index in sites:
                newSequence.append(base)
                newStructure.append(structure)
        read = self.__class__(self.id, ''.join(newSequence),
                              ''.join(newStructure))

        return read