def canonical_request(self):
        """
        The AWS SigV4 canonical request given parameters from an HTTP request.
        This process is outlined here:
        http://docs.aws.amazon.com/general/latest/gr/sigv4-create-canonical-request.html

        The canonical request is:
            request_method + '\n' +
            canonical_uri_path + '\n' +
            canonical_query_string + '\n' +
            signed_headers + '\n' +
            sha256(body).hexdigest()
        """
        signed_headers = self.signed_headers
        header_lines = "".join(
            ["%s:%s\n" % item for item in iteritems(signed_headers)])
        header_keys = ";".join([key for key in iterkeys(self.signed_headers)])
        
        return (self.request_method + "\n" +
                self.canonical_uri_path + "\n" +
                self.canonical_query_string + "\n" +
                header_lines + "\n" +
                header_keys + "\n" +
                sha256(self.body).hexdigest())