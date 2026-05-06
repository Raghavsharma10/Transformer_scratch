def buscar_por_ip_ambiente(self, ip, id_environment):
        """Get IP with an associated environment.

        :param ip: IP address in the format x1.x2.x3.x4.
        :param id_environment: Identifier of the environment. Integer value and greater than zero.

        :return: Dictionary with the following structure:

        ::

            {'ip': {'id': < id >,
            'id_vlan': < id_vlan >,
            'oct4': < oct4 >,
            'oct3': < oct3 >,
            'oct2': < oct2 >,
            'oct1': < oct1 >,
            'descricao': < descricao > }}

        :raise IpNaoExisteError: IP is not registered or not associated with environment.
        :raise InvalidParameterError: The environment identifier and/or IP is/are null or invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        """

        if not is_valid_int_param(id_environment):
            raise InvalidParameterError(
                u'Environment identifier is invalid or was not informed.')

        if not is_valid_ip(ip):
            raise InvalidParameterError(u'IP is invalid or was not informed.')

        url = 'ip/' + str(ip) + '/ambiente/' + str(id_environment) + '/'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)