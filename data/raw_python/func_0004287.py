def remove_networks(self, ids):
        """
        Set column 'active = 0' in tables redeipv4 and redeipv6]

        :param ids: ID for NetworkIPv4 and/or NetworkIPv6

        :return: Nothing

        :raise NetworkInactiveError: Unable to remove the network because it is inactive.
        :raise InvalidParameterError: Invalid ID for Network or NetworkType.
        :raise NetworkIPv4NotFoundError: NetworkIPv4 not found.
        :raise NetworkIPv6NotFoundError: NetworkIPv6 not found.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        network_map = dict()
        network_map['ids'] = ids

        code, xml = self.submit(
            {'network': network_map}, 'PUT', 'network/remove/')

        return self.response(code, xml)