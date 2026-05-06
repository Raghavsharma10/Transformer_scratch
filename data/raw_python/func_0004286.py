def deallocate_network_ipv6(self, id_network_ipv6):
        """
        Deallocate all relationships between NetworkIPv6.

        :param id_network_ipv6: ID for NetworkIPv6

        :return: Nothing

        :raise InvalidParameterError: Invalid ID for NetworkIPv6.
        :raise NetworkIPv6NotFoundError: NetworkIPv6 not found.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_network_ipv6):
            raise InvalidParameterError(
                u'The identifier of NetworkIPv6 is invalid or was not informed.')

        url = 'network/ipv6/' + str(id_network_ipv6) + '/deallocate/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)