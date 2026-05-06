def popitem(self):
        """Pops the first (key,val)"""
        sequence = (self.scalars + self.sections)
        if not sequence:
            raise KeyError(": 'popitem(): dictionary is empty'")
        key = sequence[0]
        val =  self[key]
        del self[key]
        return key, val