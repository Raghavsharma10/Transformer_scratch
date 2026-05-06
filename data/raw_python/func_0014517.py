def connect_host(self, host, volume, **kwargs):
        """Create a connection between a host and a volume.

        :param host: Name of host to connect to volume.
        :type host: str
        :param volume: Name of volume to connect to host.
        :type volume: str
        :param \*\*kwargs: See the REST API Guide on your array for the
                           documentation on the request:
                           **POST host/:host/volume/:volume**
        :type \*\*kwargs: optional

        :returns: A dictionary describing the connection between the host and volume.
        :rtype: ResponseDict

        """
        return self._request(
            "POST", "host/{0}/volume/{1}".format(host, volume), kwargs)