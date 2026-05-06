def read_name(self):
        """Reads a domain name from the packet"""
        result = ''
        off = self.offset
        next = -1
        first = off

        while 1:
            len = ord(self.data[off])
            off += 1
            if len == 0:
                break
            t = len & 0xC0
            if t == 0x00:
                result = ''.join((result, self.read_utf(off, len) + '.'))
                off += len
            elif t == 0xC0:
                if next < 0:
                    next = off + 1
                off = ((len & 0x3F) << 8) | ord(self.data[off])
                if off >= first:
                    raise Exception(
                            "Bad domain name (circular) at " + str(off))
                first = off
            else:
                raise Exception("Bad domain name at " + str(off))

        if next >= 0:
            self.offset = next
        else:
            self.offset = off

        return result