def _expand_des_key(key):
        """
        Expand the key from a 7-byte password key into a 8-byte DES key
        """
        key = key[:7] + b'\0' * (7 - len(key))
        byte = struct.unpack_from('BBBBBBB', key)
        s  = struct.pack('B', ((byte[0] >> 1) & 0x7f) << 1)
        s += struct.pack("B", ((byte[0] & 0x01) << 6 | ((byte[1] >> 2) & 0x3f)) << 1)
        s += struct.pack("B", ((byte[1] & 0x03) << 5 | ((byte[2] >> 3) & 0x1f)) << 1)
        s += struct.pack("B", ((byte[2] & 0x07) << 4 | ((byte[3] >> 4) & 0x0f)) << 1)
        s += struct.pack("B", ((byte[3] & 0x0f) << 3 | ((byte[4] >> 5) & 0x07)) << 1)
        s += struct.pack("B", ((byte[4] & 0x1f) << 2 | ((byte[5] >> 6) & 0x03)) << 1)
        s += struct.pack("B", ((byte[5] & 0x3f) << 1 | ((byte[6] >> 7) & 0x01)) << 1)
        s += struct.pack("B", (byte[6] & 0x7f) << 1)
        return s