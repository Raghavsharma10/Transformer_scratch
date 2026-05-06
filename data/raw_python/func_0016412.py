def write(self, path, data, offset=0, timeout=0):
        """write data at path

        path is a string, data binary; it is responsability of the caller
        ensure proper encoding.
        """

        # fixme: check of path type delayed to str2bytez
        if not isinstance(data, (bytes, bytearray, )):
            raise TypeError("'data' argument must be binary")

        ret, rdata = self.sendmess(MSG_WRITE, str2bytez(path) + data,
                                   size=len(data), offset=offset,
                                   timeout=timeout)
        assert not rdata, (ret, rdata)
        if ret < 0:
            raise OwnetError(-ret, self.errmess[-ret], path)