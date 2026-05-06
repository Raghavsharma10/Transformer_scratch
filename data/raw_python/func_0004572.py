def find_ip6_by_id(self, id_ip):
        """
        Get an IP6 by ID

        :param id_ip: IP6 identifier. Integer value and greater than zero.

        :return: Dictionary with the following structure:

        ::

            {'ip': {'id': < id >,
            'block1': <block1>,
            'block2': <block2>,
            'block3': <block3>,
            'block4': <block4>,
            'block5': <block5>,
            'block6': <block6>,
            'block7': <block7>,
            'block8': <block8>,
            'descricao': < description >,
            'equipamento': [ { all name equipamentos related} ], }}


        :raise IpNotAvailableError: Network dont have available IPv6.
        :raise NetworkIPv4NotFoundError: Network was not found.
        :raise UserNotAuthorizedError: User dont have permission to perform operation.
        :raise InvalidParameterError: IPv6 identifier is none or invalid.
        :raise XMLError: Networkapi failed to generate the XML response.
        :raise DataBaseError: Networkapi failed to access the database.

        """

        if not is_valid_int_param(id_ip):
            raise InvalidParameterError(
                u'Ipv6 identifier is invalid or was not informed.')

        url = 'ipv6/get/' + str(id_ip) + "/"

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)