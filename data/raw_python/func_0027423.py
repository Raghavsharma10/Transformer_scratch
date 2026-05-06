def fromhexstring(cls, hexstring):
        """
        Construct BitMap from hex string
        """
        bitstring = format(int(hexstring, 16), "0" + str(len(hexstring)/4) + "b")
        return cls.fromstring(bitstring)