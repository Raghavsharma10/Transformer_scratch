def _read_header(self):
        '''
        Little-endian
        |... 4 bytes unsigned int ...|... 4 bytes unsigned int ...|
        |        frames count        |      dimensions count      |
        '''
        self._fh.seek(0)
        buf = self._fh.read(4*2)
        fc, dc = struct.unpack("<II", buf)
        return fc, dc