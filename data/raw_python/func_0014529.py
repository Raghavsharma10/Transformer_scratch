def create_pgroup_snapshots(self, sources, **kwargs):
        """Create snapshots of pgroups from specified sources.

        :param sources: Names of pgroups of which to take snapshots.
        :type sources: list of str
        :param \*\*kwargs: See the REST API Guide on your array for the
                           documentation on the request:
                           **POST pgroup**
        :type \*\*kwargs: optional

        :returns: A list of dictionaries describing the created snapshots.
        :rtype: ResponseList

        .. note::

            Requires use of REST API 1.2 or later.

        """
        data = {"source": sources, "snap": True}
        data.update(kwargs)
        return self._request("POST", "pgroup", data)