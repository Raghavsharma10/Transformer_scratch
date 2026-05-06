def connect_array(self, address, connection_key, connection_type, **kwargs):
        """Connect this array with another one.

        :param address: IP address or DNS name of other array.
        :type address: str
        :param connection_key: Connection key of other array.
        :type connection_key: str
        :param connection_type: Type(s) of connection desired.
        :type connection_type: list
        :param \*\*kwargs: See the REST API Guide on your array for the
                           documentation on the request:
                           **POST array/connection**
        :type \*\*kwargs: optional

        :returns: A dictionary describing the connection to the other array.
        :rtype: ResponseDict

        .. note::

            Currently, the only type of connection is "replication".

        .. note::

            Requires use of REST API 1.2 or later.

        """
        data = {"management_address": address,
                "connection_key": connection_key,
                "type": connection_type}
        data.update(kwargs)
        return self._request("POST", "array/connection", data)