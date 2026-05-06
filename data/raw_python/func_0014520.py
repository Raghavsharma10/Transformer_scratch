def create_subnet(self, subnet, prefix, **kwargs):
        """Create a subnet.

        :param subnet: Name of subnet to be created.
        :type subnet: str
        :param prefix: Routing prefix of subnet to be created.
        :type prefix: str
        :param \*\*kwargs: See the REST API Guide on your array for the
                           documentation on the request:
                           **POST subnet/:subnet**
        :type \*\*kwargs: optional

        :returns: A dictionary describing the created subnet.
        :rtype: ResponseDict

        .. note::

            prefix should be specified as an IPv4 CIDR address.
            ("xxx.xxx.xxx.xxx/nn", representing prefix and prefix length)

        .. note::

            Requires use of REST API 1.5 or later.

        """
        data = {"prefix": prefix}
        data.update(kwargs)
        return self._request("POST", "subnet/{0}".format(subnet), data)