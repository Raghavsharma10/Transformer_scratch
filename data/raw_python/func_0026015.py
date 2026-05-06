def add(self, key, item):
        """Add a new key/item pair to the dictionary.  Resets an existing
        key value only if this is an exact match to a known key."""
        mmkeys = self.mmkeys
        if mmkeys is not None and not (key in self.data):
            # add abbreviations as short as minkeylength
            # always add at least one entry (even for key="")
            lenkey = len(key)
            start = min(self.minkeylength,lenkey)
            # cache references to speed up loop a bit
            mmkeysGet = mmkeys.setdefault
            for i in range(start,lenkey+1):
                mmkeysGet(key[0:i],[]).append(key)
        self.data[key] = item