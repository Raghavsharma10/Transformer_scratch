def from_buffer(self, buf):
        """
        Identify the contents of `buf`
        """
        with self.lock:
            try:
                # if we're on python3, convert buf to bytes
                # otherwise this string is passed as wchar*
                # which is not what libmagic expects
                if isinstance(buf, str) and str != bytes:
                    buf = buf.encode('utf-8', errors='replace')
                return maybe_decode(magic_buffer(self.cookie, buf))
            except MagicException as e:
                return self._handle509Bug(e)