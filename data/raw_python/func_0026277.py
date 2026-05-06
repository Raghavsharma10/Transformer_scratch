def _parse_parmlist(self, parmlist):
        """
        Parses a PARMLIST string into a dictionary of name and value 
        pairs. The parsing is complicated by the following:
        
         - parameter keynames may or may not include a length 
           specification
         - delimiter characters (=, &) may appear inside parameter
           values, provided the parameter has an explicit length.
        
        For example, the following parmlist values are possible:
        
          A=B&C=D
          A[1]=B&C[1]=D
          A=B&C[1]=D
          A[3]=B&B&C[1]=D  (Here, the value of A is "B&B")
          A[1]=B&C[3]=D=7  (Here, the value of C is "D=7")
          
        """
        parmlist = "&" + parmlist
        name_re = re.compile(r'\&([A-Z0-9_]+)(\[\d+\])?=')
        
        results = {}
        offset = 0
        match = name_re.search(parmlist, offset)
        while match:
            name, len_suffix = match.groups()
            offset = match.end()
            if len_suffix:
                val_len = int(len_suffix[1:-1])
            else:
                next_match = name_re.search(parmlist, offset)
                if next_match:
                    val_len = next_match.start() - match.end()
                else:
                    # At end of parmlist
                    val_len = len(parmlist) - match.end()
            value = parmlist[match.end() : match.end() + val_len]
            results[name.lower()] = value
                                    
            match = name_re.search(parmlist, offset)
        return results