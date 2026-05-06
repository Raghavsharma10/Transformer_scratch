def open(self, pathobj):
        """
        Opens the remote file and returns a file-like object HTTPResponse
        Given the nature of HTTP streaming, this object doesn't support
        seek()
        """
        url = str(pathobj)
        raw, code = self.rest_get_stream(url, auth=pathobj.auth, verify=pathobj.verify,
                                         cert=pathobj.cert)

        if not code == 200:
            raise RuntimeError("%d" % code)

        return raw