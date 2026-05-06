def join(self, iterable):
        """Joins an iterable yielding strings or FmtStrs with self as separator"""
        before = []
        chunks = []
        for i, s in enumerate(iterable):
            chunks.extend(before)
            before = self.chunks
            if isinstance(s, FmtStr):
                chunks.extend(s.chunks)
            elif isinstance(s, (bytes, unicode)):
                chunks.extend(fmtstr(s).chunks) #TODO just make a chunk directly
            else:
                raise TypeError("expected str or FmtStr, %r found" % type(s))
        return FmtStr(*chunks)