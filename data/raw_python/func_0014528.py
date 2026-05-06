def send_pgroup_snapshot(self, source, **kwargs):
        """ Send an existing pgroup snapshot to target(s)

        :param source: Name of pgroup snapshot to send.
        :type source: str
        :param \*\*kwargs: See the REST API Guide on your array for the
                           documentation on the request:
                           **POST pgroup**
        :type \*\*kwargs: optional

        :returns: A list of dictionaries describing the sent snapshots.
        :rtype: ResponseList

        .. note::

            Requires use of REST API 1.16 or later.

        """
        data = {"name": [source], "action":"send"}
        data.update(kwargs)
        return self._request("POST", "pgroup", data)