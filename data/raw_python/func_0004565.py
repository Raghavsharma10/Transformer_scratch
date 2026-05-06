def get_available_ip6(self, id_network6):
        """
        Get a available IP in Network ipv6

        :param id_network6: Network ipv6 identifier. Integer value and greater than zero.

        :return: Dictionary with the following structure:

        ::

            {'ip6': {'ip6': < available_ip6 >}}

        :raise IpNotAvailableError: Network dont have available IP.
        :raise NetworkIPv4NotFoundError: Network was not found.
        :raise UserNotAuthorizedError: User dont have permission to get a available IP.
        :raise InvalidParameterError: Network ipv6 identifier is none or invalid.
        :raise XMLError: Networkapi failed to generate the XML response.
        :raise DataBaseError: Networkapi failed to access the database.

        """

        if not is_valid_int_param(id_network6):
            raise InvalidParameterError(
                u'Network ipv6 identifier is invalid or was not informed.')

        url = 'ip/availableip6/' + str(id_network6) + "/"

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)