def findInvariantPartitioning(self):
        """Keep the initial ordering of the symmetry orders
        but make all values unique.  For example, if there are
        two symmetry orders equal to 0, convert them to 0 and 1
        and add 1 to the remaining orders

          [0, 1, 0, 1]
        should become
          [0, 2, 1, 3]"""
        
        symorders = self.symorders[:]
        _range = range(len(symorders))
        while 1:
            pos = self.findLowest(symorders)
            if pos == -1:
                self.symorders = symorders
                return
            for i in _range:
                symorders[i] = symorders[i] * 2 + 1
            symorders[pos] = symorders[pos] - 1
            
            symorders = self.findInvariant(symorders)