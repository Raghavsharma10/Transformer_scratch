def create_vlan_interface(self, interface, subnet, **kwargs):
        """Create a vlan interface

        :param interface: Name of interface to be created.
        :type interface: str
        :param subnet: Subnet associated with interface to be created
        :type subnet: str
        :param \*\*kwargs: See the REST API Guide on your array for the
                           documentation on the request:
                           **POST network/vif/:vlan_interface**
        :type \*\*kwargs: optional

        :returns: A dictionary describing the created interface
        :rtype: ResponseDict

        .. note::

            Requires use of REST API 1.5 or later.

        """
        data = {"subnet": subnet}
        data.update(kwargs)
        return self._request("POST", "network/vif/{0}".format(interface), data)