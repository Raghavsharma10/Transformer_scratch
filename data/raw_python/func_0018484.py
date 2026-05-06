def knearest(self, f_or_start, end=None, chrom=None, k=1):
        """return the n nearest neighbors to the given feature
        f: a Feature object
        k: the number of features to return
        """


        if end is not None:
            f = Feature(f_or_start, end, chrom=chrom)
        else:
            f = f_or_start

        DIST = 2000
        feats = filter_feats(self.find(f.start - DIST, f.end + DIST, chrom=f.chrom), f, k)
        if len(feats) >= k:
            return feats

        nfeats = k - len(feats)
        fleft = Feature(f.start - DIST, f.start, chrom=f.chrom)
        feats.extend(self.left(fleft, n=nfeats))

        fright = Feature(f.end, f.end + DIST, chrom=f.chrom)
        feats.extend(self.right(fright, n=nfeats))
        return filter_feats(feats, f, k)