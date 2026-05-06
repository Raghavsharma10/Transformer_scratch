def get_offload(self, name, **kwargs):
        """Return a dictionary describing the connected offload target.

        :param offload: Name of offload target to get information about.
        :type offload: str
        :param \*\*kwargs: See the REST API Guide on your array for the
                           documentation on the request:
                           **GET offload/::offload**
        :type \*\*kwargs: optional

        :returns: A dictionary describing the offload connection.
        :rtype: ResponseDict

        """
        # Unbox if a list to accommodate a bug in REST 1.14
        result = self._request("GET", "offload/{0}".format(name), kwargs)
        if isinstance(result, list):
            headers = result.headers
            result = ResponseDict(result[0])
            result.headers = headers
        return result