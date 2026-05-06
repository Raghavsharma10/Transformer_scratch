def fromone(cls, index, pad=0, autostep=None):
        """Class method that returns a new RangeSet of one single item or
        a single range (from integer or slice object)."""
        inst = RangeSet(autostep=autostep)
        # support slice object with duck-typing
        try:
            inst.add(index, pad)
        except TypeError:
            if not index.stop:
                raise ValueError("Invalid range upper limit (%s)" % index.stop)
            inst.add_range(index.start or 0, index.stop, index.step or 1, pad)
        return inst