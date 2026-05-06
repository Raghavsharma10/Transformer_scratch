def pack_chunk(tag, data):
        """Pack a PNG chunk for serializing to disk"""
        to_check = tag + data
        return (struct.pack(b"!I", len(data)) + to_check +
                struct.pack(b"!I", zlib.crc32(to_check) & 0xFFFFFFFF))