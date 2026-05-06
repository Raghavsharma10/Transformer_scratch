def compare(self, otherdigest, ishex=False):
        """Compute difference in bits between own digest and another.
           returns -127 to 128; 128 is the same, -127 is different"""
        bits = 0
        myd = self.digest()
        if ishex:
            # convert to 32-tuple of unsighed two-byte INTs
            otherdigest = tuple([int(otherdigest[i:i+2],16) for i in range(0,63,2)])
        for i in range(32):
            bits += POPC[255 & myd[i] ^ otherdigest[i]]
        return 128 - bits