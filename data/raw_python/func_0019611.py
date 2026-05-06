def get_size(self, chrom=None):
        """ Return the sizes of all sequences in the index, or the size of chrom if specified
        as an optional argument """
        if len(self.size) == 0:
            raise LookupError("no chromosomes in index, is the index correct?")

        if chrom:
            if chrom in self.size:
                return self.size[chrom]
            else: 
                raise KeyError("chromosome {} not in index".format(chrom))
        total = 0
        for size in self.size.values():
            total += size

        return total