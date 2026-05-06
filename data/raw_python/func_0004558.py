def get_ipv4(self, id_ip):
        """Get IPv4 by id.

        :param id_ip: ID of IPv4.

        :return: Dictionary with the following structure:

        ::

            {'ip': {'id': < id >,
            'networkipv4': < networkipv4 >,
            'oct4': < oct4 >,
            'oct3': < oct3 >,
            'oct2': < oct2 >,
            'oct1': < oct1 >,
            'descricao': < descricao >,
            'equipamentos': [ { all name of equipments related } ] , }}

        :raise IpNaoExisteError: IP is not registered.
        :raise InvalidParameterError: IP identifier is null or invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_ip):
            raise InvalidParameterError(
                u'The IPv4 identifier is invalid or was not informed.')

        url = 'ip/get-ipv4/' + str(id_ip) + '/'

        code, xml = self.submit(None, 'GET', url)

        key = 'ipv4'
        return get_list_map(self.response(code, xml, ["equipamentos"]), key)