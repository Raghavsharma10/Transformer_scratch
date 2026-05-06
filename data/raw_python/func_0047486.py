def slice_sequence(self,start,end):
     """Slice the mapping by the position in the sequence
        
        First coordinate is 0-indexed start
        Second coordinate is 1-indexed finish

     """
     #find the sequence length
     l = self.length
     indexstart = start
     indexend = end
     ns = []
     tot = 0
     for r in self._rngs:
        tot += r.length
        n = r.copy()
        if indexstart > r.length:  
           indexstart-=r.length
           continue
        n.start = n.start+indexstart
        if tot > end: 
           diff = tot-end
           n.end -= diff
           tot = end
        indexstart = 0
        ns.append(n)
        if tot == end: break
     if len(ns)==0: return None
     return MappingGeneric(ns,self._options)