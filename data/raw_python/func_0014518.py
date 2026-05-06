def connect_hgroup(self, hgroup, volume, **kwargs):
        """Create a shared connection between a host group and a volume.

        :param hgroup: Name of hgroup to connect to volume.
        :type hgroup: str
        :param volume: Name of volume to connect to hgroup.
        :type volume: str
        :param \*\*kwargs: See the REST API Guide on your array for the
                           documentation on the request:
                           **POST hgroup/:hgroup/volume/:volume**
        :type \*\*kwargs: optional

        :returns: A dictionary describing the connection between the hgroup and volume.
        :rtype: ResponseDict

        """
        return self._request(
            "POST", "hgroup/{0}/volume/{1}".format(hgroup, volume), kwargs)