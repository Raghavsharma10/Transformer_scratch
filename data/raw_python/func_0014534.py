def list_certificates(self):
        """Get the attributes of the current array certificate.

        :param \*\*kwargs: See the REST API Guide on your array for the
                           documentation on the request:
                           **GET cert**
        :type \*\*kwargs: optional

        :returns: A list of dictionaries describing all configured certificates.
        :rtype: ResponseList

        .. note::

            Requires use of REST API 1.12 or later.

        """

        # This call takes no parameters.
        if self._rest_version >= LooseVersion("1.12"):
            return self._request("GET", "cert")
        else:
            # If someone tries to call this against a too-early api version,
            # do the best we can to provide expected behavior.
            cert = self._request("GET", "cert")
            out = ResponseList([cert])
            out.headers = cert.headers
            return out