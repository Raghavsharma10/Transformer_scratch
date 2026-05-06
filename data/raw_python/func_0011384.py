def do_peek(self, args):
        """Peek at the next 16 bytes in the stream::

        Example:

            The peek command will display the next 16 hex bytes in the input
            stream::
                pfp> peek
                89 50 4e 47 0d 0a 1a 0a 00 00 00 0d 49 48 44 52 .PNG........IHDR
        """
        s = self._interp._stream
        # make a copy of it
        pos = s.tell()
        saved_bits = collections.deque(s._bits)
        data = s.read(0x10)
        s.seek(pos, 0)
        s._bits = saved_bits

        parts = ["{:02x}".format(ord(data[x:x+1])) for x in range(len(data))]
        if len(parts) != 0x10:
            parts += ["  "] * (0x10 - len(parts))
        hex_line = " ".join(parts)

        res = utils.binary("")
        for x in range(len(data)):
            char = data[x:x+1]
            val = ord(char)
            if 0x20 <= val <= 0x7e:
                res += char
            else:
                res += utils.binary(".")
        
        if len(res) < 0x10:
            res += utils.binary(" " * (0x10 - len(res)))

        res = "{} {}".format(hex_line, utils.string(res))
        if len(saved_bits) > 0:
            reverse_bits = reversed(list(saved_bits))
            print("bits: {}".format(" ".join(str(x) for x in reverse_bits)))
        print(res)