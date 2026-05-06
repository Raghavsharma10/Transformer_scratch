def signed_headers(self):
        """
        An ordered dictionary containing the signed header names and values.
        """
        # See if the signed headers are listed in the query string
        signed_headers = self.query_parameters.get(_x_amz_signedheaders)
        if signed_headers is not None:
            signed_headers = url_unquote(signed_headers[0])
        else:
            # Get this from the authentication header
            signed_headers = self.authorization_header_parameters[
                _signedheaders]

        # Header names are separated by semicolons.
        parts = signed_headers.split(";")

        # Make sure the signed headers list is canonicalized.  For security
        # reasons, we consider it an error if it isn't.
        canonicalized = sorted([sh.lower() for sh in parts])
        if parts != canonicalized:
            raise AttributeError("SignedHeaders is not canonicalized: %r" %
                                 (signed_headers,))

        # Allow iteration in-order.
        return OrderedDict([(header, self.headers[header])
                            for header in signed_headers.split(";")])