def get_available_ip4(self, id_network):
        """
        Get a available IP in the network ipv4

        :param id_network: Network identifier. Integer value and greater than zero.

        :return: Dictionary with the following structure:

        ::

            {'ip': {'ip': < available_ip >}}

        :raise IpNotAvailableError: Network dont have available IP for insert a new IP
        :raise NetworkIPv4NotFoundError: Network is not found
        :raise UserNotAuthorizedError: User dont have permission to get a available IP
        :raise InvalidParameterError: Network identifier is null or invalid.
        :raise XMLError: Networkapi failed to generate the XML response.
        :raise DataBaseError: Networkapi failed to access the database.
        """

        if not is_valid_int_param(id_network):
            raise InvalidParameterError(
                u'Network identifier is invalid or was not informed.')

        url = 'ip/availableip4/' + str(id_network) + "/"

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)