def seek(self, offset, whence=os.SEEK_SET):
        """Seek to position in stream, see file.seek"""

        pos = None
        if whence == os.SEEK_SET:
            pos = self.offset + offset
        elif whence == os.SEEK_CUR:
            pos = self.tell() + offset
        elif whence == os.SEEK_END:
            pos = self.offset + self.len + offset
        else:
            raise ValueError("invalid whence {}".format(whence))

        if pos > self.offset + self.len or pos < self.offset:
            raise ValueError("seek position beyond chunk area")

        self.parent_fd.seek(pos, os.SEEK_SET)