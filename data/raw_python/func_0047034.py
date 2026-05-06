def tags(self):
     """Access the auxillary data here"""
     if self._tags: return self._tags
     tags = {}
     if not tags: return {}
     for m in [[y.group(1),y.group(2),y.group(3)] for y in [re.match('([^:]{2,2}):([^:]):(.+)$',x) for x in self.entries.optional_fields.split("\t")]]:
        if m[1] == 'i': m[2] = int(m[2])
        elif m[1] == 'f': m[2] = float(m[2])
        tags[m[0]] = TAGDatum(m[1],m[2])
     self._tags = tags
     return self._tags