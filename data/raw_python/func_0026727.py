def _ordered(self, odict):
        """Convert the object into a plain OrderedDict."""
        ndict = OrderedDict()

        if isinstance(odict, CatDict) or isinstance(odict, Entry):
            key = odict.sort_func
        else:
            key = None

        nkeys = list(sorted(odict.keys(), key=key))
        for key in nkeys:
            if isinstance(odict[key], OrderedDict):
                odict[key] = self._ordered(odict[key])
            if isinstance(odict[key], list):
                if (not (odict[key] and
                         not isinstance(odict[key][0], OrderedDict))):
                    nlist = []
                    for item in odict[key]:
                        if isinstance(item, OrderedDict):
                            nlist.append(self._ordered(item))
                        else:
                            nlist.append(item)
                    odict[key] = nlist
            ndict[key] = odict[key]

        return ndict