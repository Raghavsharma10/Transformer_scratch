def bed12(self, score="0", rgb="."):
        """
        return a bed12 (http://genome.ucsc.edu/FAQ/FAQformat.html#format1)
        representation of this interval
        """
        if not self.is_gene_pred:
            raise CruzException("can't create bed12 from non genepred feature")
        exons = list(self.exons)
        # go from global start, stop, to relative start, length...
        sizes = ",".join([str(e[1] - e[0]) for e in exons]) + ","
        starts = ",".join([str(e[0] - self.txStart) for e in exons]) + ","
        name = self.name2 + "," + self.name if hasattr(self, "name2") \
                                            else self.name
        return "\t".join(map(str, (
            self.chrom, self.txStart, self.txEnd, name,
            score, self.strand, self.cdsStart, self.cdsEnd, rgb,
            len(exons), sizes, starts)))