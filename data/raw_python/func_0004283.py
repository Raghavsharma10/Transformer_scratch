def deallocate_network_ipv4(self, id_network_ipv4):
        """
        Deallocate all relationships between NetworkIPv4.

        :param id_network_ipv4: ID for NetworkIPv4

        :return: Nothing

        :raise InvalidParameterError: Invalid ID for NetworkIPv4.
        :raise NetworkIPv4NotFoundError: NetworkIPv4 not found.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_network_ipv4):
            raise InvalidParameterError(
                u'The identifier of NetworkIPv4 is invalid or was not informed.')

        url = 'network/ipv4/' + str(id_network_ipv4) + '/deallocate/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)