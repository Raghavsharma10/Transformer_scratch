def _checksum(self, packet):
        '''calculate the XOR checksum of a packet in string format'''
        xorsum = 0
        for s in packet:
            xorsum ^= ord(s)
        return xorsum