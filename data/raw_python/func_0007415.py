def deploy(self, pathobj, fobj, md5=None, sha1=None, parameters=None):
        """
        Uploads a given file-like object
        HTTP chunked encoding will be attempted
        """
        if isinstance(fobj, urllib3.response.HTTPResponse):
            fobj = HTTPResponseWrapper(fobj)

        url = str(pathobj)

        if parameters:
            url += ";%s" % encode_matrix_parameters(parameters)

        headers = {}

        if md5:
            headers['X-Checksum-Md5'] = md5
        if sha1:
            headers['X-Checksum-Sha1'] = sha1

        text, code = self.rest_put_stream(url,
                                          fobj,
                                          headers=headers,
                                          auth=pathobj.auth,
                                          verify=pathobj.verify,
                                          cert=pathobj.cert)

        if code not in [200, 201]:
            raise RuntimeError("%s" % text)