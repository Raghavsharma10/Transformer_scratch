def find_ip6_by_network(self, id_network):
        """List IPv6 from network.

        :param id_network: Network ipv6 identifier. Integer value and greater than zero.

        :return: Dictionary with the following structure:

        ::

            {'ip': {'id': < id >,
            'id_vlan': < id_vlan >,
            'block1': <block1>,
            'block2': <block2>,
            'block3': <block3>,
            'block4': <block4>,
            'block5': <block5>,
            'block6': <block6>,
            'block7': <block7>,
            'block8': <block8>,
            'descricao': < description >
            'equipamento': [ { all name equipamentos related } ], }}

        :raise IpNaoExisteError: Network does not have any ips.
        :raise InvalidParameterError: Network identifier is none or invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        """

        if not is_valid_int_param(id_network):
            raise InvalidParameterError(
                u'Network identifier is invalid or was not informed.')

        url = 'ip/id_network_ipv6/' + str(id_network) + "/"

        code, xml = self.submit(None, 'GET', url)

        key = "ips"
        return get_list_map(self.response(code, xml, [key]), key)