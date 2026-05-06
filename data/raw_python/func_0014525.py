def create_snmp_manager(self, manager, host, **kwargs):
        """Create an SNMP manager.

        :param manager: Name of manager to be created.
        :type manager: str
        :param host: IP address or DNS name of SNMP server to be used.
        :type host: str
        :param \*\*kwargs: See the REST API Guide on your array for the
                           documentation on the request:
                           **POST snmp/:manager**
        :type \*\*kwargs: optional

        :returns: A dictionary describing the created SNMP manager.
        :rtype: ResponseDict

        """
        data = {"host": host}
        data.update(kwargs)
        return self._request("POST", "snmp/{0}".format(manager), data)