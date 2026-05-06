def downstream(self, f, n=1):
        """find n downstream features where downstream is determined by
        the strand of the query Feature f
        Overlapping features are not considered.

        f: a Feature object
        n: the number of features to return
        """
        if f.strand == -1:
            return self.left(f, n)
        return self.right(f, n)