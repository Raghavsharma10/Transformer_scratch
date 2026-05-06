def chunks(f):
        """Split read PNG image data into chunks"""
        while 1:
            try:
                length = struct.unpack(b"!I", f.read(4))[0]
                tag = f.read(4)
                data = f.read(length)
                crc = struct.unpack(b"!I", f.read(4))[0]
            except struct.error:
                return
            if zlib.crc32(tag + data) & 0xFFFFFFFF != crc:
                raise IOError('Checksum fail')
            yield tag, data