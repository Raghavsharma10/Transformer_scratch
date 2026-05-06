def _build_inheritance_chain(cls, bases, *names, merge=False):
        """For all of the names build a ChainMap containing a map for every
        base class."""
        result = []
        for name in names:
            maps = []
            for base in bases:
                bmap = getattr(base, name, None)
                if bmap is not None:
                    assert isinstance(bmap, (dict, ChainMap))
                    if len(bmap):
                        if isinstance(bmap, ChainMap):
                            maps.extend(bmap.maps)
                        else:
                            maps.append(bmap)
            result.append(ChainMap({}, *maps))
        if merge:
            result = [dict(map) for map in result]
        if len(names) == 1:
            return result[0]
        return result