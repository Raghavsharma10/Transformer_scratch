def set_properties(self, pathobj, props, recursive):
        """
        Set artifact properties
        """
        url = '/'.join([pathobj.drive,
                        'api/storage',
                        str(pathobj.relative_to(pathobj.drive)).strip('/')])

        params = {'properties': encode_properties(props)}

        if not recursive:
            params['recursive'] = '0'

        text, code = self.rest_put(url,
                                   params=params,
                                   auth=pathobj.auth,
                                   verify=pathobj.verify,
                                   cert=pathobj.cert)

        if code == 404 and "Unable to find item" in text:
            raise OSError(2, "No such file or directory: '%s'" % url)
        if code != 204:
            raise RuntimeError(text)