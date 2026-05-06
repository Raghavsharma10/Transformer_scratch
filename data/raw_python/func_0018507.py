def exons(self):
        """
        return a list of exons [(start, stop)] for this object if appropriate
        """
        # drop the trailing comma
        if not self.is_gene_pred: return []
        if hasattr(self, "exonStarts"):
            try:
                starts = (long(s) for s in self.exonStarts[:-1].split(","))
                ends = (long(s) for s in self.exonEnds[:-1].split(","))
            except TypeError:
                starts = (long(s) for s in self.exonStarts[:-1].decode().split(","))
                ends = (long(s) for s in self.exonEnds[:-1].decode().split(","))
        else: # it is bed12
            starts = [self.start + long(s) for s in
                        self.chromStarts[:-1].decode().split(",")]
            ends = [starts[i] + long(size) for i, size \
                        in enumerate(self.blockSizes[:-1].decode().split(","))]


        return zip(starts, ends)