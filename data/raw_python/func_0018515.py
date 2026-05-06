def upstream(self, distance):
        """
        return the (start, end) of the region before the geneStart
        """
        if getattr(self, "strand", None) == "+":
            e = self.start
            s = e - distance
        else:
            s = self.end
            e = s + distance
        return self._xstream(s, e)