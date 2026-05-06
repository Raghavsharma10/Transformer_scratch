def upstream(self, table, chrom_or_feat, start=None, end=None, k=1):
        """
        Return k-nearest upstream features

        Parameters
        ----------

        table : str or table
            table against which to query

        chrom_or_feat : str or feat
            either a chromosome, e.g. 'chr3' or a feature with .chrom, .start,
            .end attributes

        start : int
            if `chrom_or_feat` is a chrom, then this must be the integer start

        end : int
            if `chrom_or_feat` is a chrom, then this must be the integer end

        k : int
            number of upstream neighbors to return
        """
        res = self.knearest(table, chrom_or_feat, start, end, k, "up")
        end = getattr(chrom_or_feat, "end", end)
        start = getattr(chrom_or_feat, "start", start)
        rev = getattr(chrom_or_feat, "strand", "+") == "-"
        if rev:
            return [x for x in res if x.end > start]
        else:
            return [x for x in res if x.start < end]