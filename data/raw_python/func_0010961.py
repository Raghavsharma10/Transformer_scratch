def from_def(cls, obj):
        """ Builds a profile object from a raw player summary object """
        prof = cls(obj["steamid"])
        prof._cache = obj

        return prof