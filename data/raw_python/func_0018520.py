def bed(self, *attrs, **kwargs):
        """
        return a bed formatted string of this feature
        """
        exclude = ("chrom", "start", "end", "txStart", "txEnd", "chromStart",
                "chromEnd")
        if self.is_gene_pred:
            return self.bed12(**kwargs)
        return "\t".join(map(str, (
                 [self.chrom, self.start, self.end] +
                 [getattr(self, attr) for attr in attrs if not attr in exclude]
                         )))