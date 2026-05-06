def _mmInit(self):
        """Create the minimum match dictionary of keys"""
        # cache references to speed up loop a bit
        mmkeys = {}
        mmkeysGet = mmkeys.setdefault
        minkeylength = self.minkeylength
        for key in self.data.keys():
            # add abbreviations as short as minkeylength
            # always add at least one entry (even for key="")
            lenkey = len(key)
            start = min(minkeylength,lenkey)
            for i in range(start,lenkey+1):
                mmkeysGet(key[0:i],[]).append(key)
        self.mmkeys = mmkeys