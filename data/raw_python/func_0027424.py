def fromstring(cls, bitstring):
        """
        Construct BitMap from string
        """
        nbits = len(bitstring)
        bm = cls(nbits)
        for i in xrange(nbits):
            if bitstring[-i-1] == '1':
                bm.set(i)
            elif bitstring[-i-1] != '0':
                raise Exception("Invalid bit string!")
        return bm