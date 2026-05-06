def knearest(self, table, chrom_or_feat, start=None, end=None, k=1,
            _direction=None):
        """
        Return k-nearest features

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
            number of downstream neighbors to return

        _direction : (None, "up", "down")
            internal (don't use this)
        """
        assert _direction in (None, "up", "down")

        # they sent in a feature
        if start is None:
            assert end is None
            chrom, start, end = chrom_or_feat.chrom, chrom_or_feat.start, chrom_or_feat.end

            # if the query is directional and the feature as a strand,
            # adjust...
            if _direction in ("up", "down") and getattr(chrom_or_feat,
                    "strand", None) == "-":
                _direction = "up" if _direction == "down" else "up"
        else:
            chrom = chrom_or_feat

        qstart, qend = long(start), long(end)
        res = self.bin_query(table, chrom, qstart, qend)

        i, change = 1, 350
        try:
            while res.count() < k:
                if _direction in (None, "up"):
                    if qstart == 0 and _direction == "up": break
                    qstart = max(0, qstart - change)
                if _direction in (None, "down"):
                    qend += change
                i += 1
                change *= (i + 5)
                res = self.bin_query(table, chrom, qstart, qend)
        except BigException:
            return []

        def dist(f):
            d = 0
            if start > f.end:
                d = start - f.end
            elif f.start > end:
                d = f.start - end
            # add dist as an attribute to the feature
            return d

        dists = sorted([(dist(f), f) for f in res])
        if len(dists) == 0:
            return []

        dists, res = zip(*dists)

        if len(res) == k:
            return res

        if k > len(res): # had to break because of end of chrom
            if k == 0: return []
            k = len(res)

        ndist = dists[k - 1]
        # include all features that are the same distance as the nth closest
        # feature (accounts for ties).
        while k < len(res) and dists[k] == ndist:
            k = k + 1
        return res[:k]