def slice_target(self,chr,start,end):
     """Slice the mapping by the target coordinate
        
        First coordinate is 0-indexed start
        Second coordinate is 1-indexed finish

     """
     # create a range that we are going to intersect with
     trng = Bed(chr,start,end)
     nrngs = []
     for r in self._rngs:
        i = r.intersect(trng)
        if not i: continue
        nrngs.append(i)
     if len(nrngs) == 0: return None
     return MappingGeneric(nrngs,self._options)