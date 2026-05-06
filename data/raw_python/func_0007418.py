def del_properties(self, pathobj, props, recursive):
        """
        Delete artifact properties
        """
        if isinstance(props, str):
            props = (props,)

        url = '/'.join([pathobj.drive,
                        'api/storage',
                        str(pathobj.relative_to(pathobj.drive)).strip('/')])

        params = {'properties': ','.join(sorted(props))}

        if not recursive:
            params['recursive'] = '0'

        text, code = self.rest_del(url,
                                   params=params,
                                   auth=pathobj.auth,
                                   verify=pathobj.verify,
                                   cert=pathobj.cert)

        if code == 404 and "Unable to find item" in text:
            raise OSError(2, "No such file or directory: '%s'" % url)
        if code != 204:
            raise RuntimeError(text)