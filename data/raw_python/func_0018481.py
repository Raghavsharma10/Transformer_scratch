def right(self, f, n=1):
        """return the nearest n features strictly to the right of a Feature f.
        Overlapping features are not considered as to the right.

        f: a Feature object
        n: the number of features to return
        """
        intervals = self.intervals[f.chrom]
        ilen = len(intervals)
        iright = binsearch_right_end(intervals, f.end, 0, ilen)
        results = []

        while iright < ilen:
            i = len(results)
            if i > n:
                if distance(f, results[i - 1]) != distance(f, results[i - 2]):
                    return results[:i - 1]
            other = intervals[iright]
            iright += 1
            if distance(other, f) == 0: continue
            results.append(other)
        return results