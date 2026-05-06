def fromlist(cls, rnglist, autostep=None):
        """Class method that returns a new RangeSet with ranges from provided
        list."""
        inst = RangeSet(autostep=autostep)
        inst.updaten(rnglist)
        return inst