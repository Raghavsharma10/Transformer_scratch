def move(self, src, dst):
        """
        Move artifact from src to dst
        """
        url = '/'.join([src.drive,
                        'api/move',
                        str(src.relative_to(src.drive)).rstrip('/')])

        params = {'to': str(dst.relative_to(dst.drive)).rstrip('/')}

        text, code = self.rest_post(url,
                                    params=params,
                                    auth=src.auth,
                                    verify=src.verify,
                                    cert=src.cert)

        if code not in [200, 201]:
            raise RuntimeError("%s" % text)