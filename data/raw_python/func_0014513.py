def create_snapshots(self, volumes, **kwargs):
        """Create snapshots of the listed volumes.

        :param volumes: List of names of the volumes to snapshot.
        :type volumes: list of str
        :param \*\*kwargs: See the REST API Guide on your array for the
                           documentation on the request:
                           **POST volume**
        :type \*\*kwargs: optional

        :returns: A list of dictionaries describing the new snapshots.
        :rtype: ResponseDict

        """
        data = {"source": volumes, "snap": True}
        data.update(kwargs)
        return self._request("POST", "volume", data)