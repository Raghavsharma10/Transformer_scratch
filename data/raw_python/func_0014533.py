def get_certificate(self, **kwargs):
        """Get the attributes of the current array certificate.

        :param \*\*kwargs: See the REST API Guide on your array for the
                           documentation on the request:
                           **GET cert**
        :type \*\*kwargs: optional

        :returns: A dictionary describing the configured array certificate.
        :rtype: ResponseDict

        .. note::

            Requires use of REST API 1.3 or later.

        """

        if self._rest_version >= LooseVersion("1.12"):
            return self._request("GET",
                "cert/{0}".format(kwargs.pop('name', 'management')), kwargs)
        else:
            return self._request("GET", "cert", kwargs)