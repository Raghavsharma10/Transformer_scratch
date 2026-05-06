def serialtoflat(self, bytes, width=None):
        """Convert serial format (byte stream) pixel data to flat row
        flat pixel.
        """

        if self.bitdepth == 8:
            return bytes
        if self.bitdepth == 16:
            bytes = tostring(bytes)
            return array('H',
              struct.unpack('!%dH' % (len(bytes)//2), bytes))
        assert self.bitdepth < 8
        if width is None:
            width = self.width
        # Samples per byte
        spb = 8//self.bitdepth
        out = array('B')
        mask = 2**self.bitdepth - 1
        shifts = map(self.bitdepth.__mul__, reversed(range(spb)))
        l = width
        for o in bytes:
            out.extend([(mask&(o>>s)) for s in shifts][:l])
            l -= spb
            if l <= 0:
                l = width
        return out