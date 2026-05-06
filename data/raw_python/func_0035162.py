def read_transfac( fin, alphabet = None) :
        """ Parse a sequence matrix from a file. 
        Returns a tuple of (alphabet, matrix)
        """
   
        items = []

        start=True
        for line in fin :
            if line.isspace() or line[0] =='#' : continue
            stuff = line.split()
            if start and stuff[0] != 'PO' and stuff[0] != 'P0': continue
            if stuff[0]=='XX' or stuff[0]=='//': break
            start = False
            items.append(stuff)
        if len(items) < 2  :
            raise ValueError("Vacuous file.")

        # Is the first line a header line?
        header = items.pop(0)
        hcols = len(header)
        rows = len(items)
        cols = len(items[0])
        if not( header[0] == 'PO' or header[0] =='P0' or hcols == cols-1 or hcols == cols-2) :
            raise ValueError("Missing header line!")

        # Do all lines (except the first) contain the same number of items?
        cols = len(items[0])
        for i in range(1, len(items)) :
            if cols != len(items[i]) :
                raise ValueError("Inconsistant length, row %d: " % i)

        # Vertical or horizontal arrangement?
        if header[0] == 'PO' or header[0] == 'P0': header.pop(0)

        position_header = True    
        alphabet_header = True    
        for h in header :
            if not corebio.utils.isint(h) : position_header = False
#allow non-alphabetic            if not str.isalpha(h) : alphabet_header = False

        if not position_header and not alphabet_header :
            raise ValueError("Can't parse header: %s" % str(header))

        if position_header and alphabet_header :
            raise ValueError("Can't parse header")


        # Check row headers
        if alphabet_header :
            for i,r in enumerate(items) :
                if not corebio.utils.isint(r[0]) and r[0][0]!='P' : 
                    raise ValueError(
                        "Expected position as first item on line %d" % i)
                r.pop(0)
                defacto_alphabet = ''.join(header)
        else :
            a = []
            for i,r in enumerate(items) :
                if not ischar(r[0]) and r[0][0]!='P' : 
                    raise ValueError(
                        "Expected position as first item on line %d" % i)
                a.append(r.pop(0))
            defacto_alphabet = ''.join(a)                

        # Check defacto_alphabet
        defacto_alphabet = corebio.seq.Alphabet(defacto_alphabet)

        if alphabet :
            if not defacto_alphabet.alphabetic(alphabet) :
                raise ValueError("Incompatible alphabets: %s , %s (defacto)"
                                 % (alphabet, defacto_alphabet))
        else :            
            alphabets = (unambiguous_rna_alphabet,
                        unambiguous_dna_alphabet,                      
                        unambiguous_protein_alphabet,
                      )
            for a in alphabets :
                if defacto_alphabet.alphabetic(a) :
                    alphabet = a
                    break
            if not alphabet :
                alphabet = defacto_alphabet
   

        # The last item of each row may be extra cruft. Remove
        if len(items[0]) == len(header) +1 :
            for r in items :
                r.pop()

        # items should now be a list of lists of numbers (as strings) 
        rows = len(items)
        cols = len(items[0])
        matrix = numpy.zeros( (rows,cols) , dtype=numpy.float64) 
        for r in range( rows) :
            for c in range(cols):
                matrix[r,c] = float( items[r][c]) 

        if position_header :
            matrix.transpose() 

        return _my_Motif(defacto_alphabet, matrix).reindex(alphabet)