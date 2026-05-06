def get_ipv6(self, id_ip):
        """Get IPv6 by id.

        :param id_ip: ID of IPv6.

        :return: Dictionary with the following structure:

        ::

            {'ip': {'id': < id >,
            'networkipv6': < networkipv6 >,
            'block1': < block1 >,
            'block2': < block2 >,
            'block3': < block3 >,
            'block4': < block4 >,
            'block5': < block5 >,
            'block6': < block6 >,
            'block7': < block7 >,
            'block8': < block8 >,
            'description': < description >,
            'equipamentos': [ { all name of equipments related } ] , }}

        :raise IpNaoExisteError: IP is not registered.
        :raise InvalidParameterError: IP identifier is null or invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_ip):
            raise InvalidParameterError(
                u'The IPv6 identifier is invalid or was not informed.')

        url = 'ip/get-ipv6/' + str(id_ip) + '/'

        code, xml = self.submit(None, 'GET', url)

        key = 'ipv6'
        return get_list_map(self.response(code, xml, ["equipamentos"]), key)