def read(self, path, size=MAX_PAYLOAD, offset=0, timeout=0):
        """read data at path"""

        if size > MAX_PAYLOAD:
            raise ValueError("size cannot exceed %d" % MAX_PAYLOAD)

        ret, data = self.sendmess(MSG_READ, str2bytez(path),
                                  size=size, offset=offset, timeout=timeout)
        if ret < 0:
            raise OwnetError(-ret, self.errmess[-ret], path)
        return data