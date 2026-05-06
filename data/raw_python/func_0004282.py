def get_network_ipv6(self, id_network):
        """
        Get networkipv6

        :param id_network: Identifier of the Network. Integer value and greater than zero.
        :return: Following dictionary:

        ::

          {'network': {'id': < id_networkIpv6 >,
          'network_type': < id_tipo_rede >,
          'ambiente_vip': < id_ambiente_viṕ >,
          'vlan': <id_vlan>
          'block1': < rede_oct1 >,
          'block2': < rede_oct2 >,
          'block3': < rede_oct3 >,
          'block4': < rede_oct4 >,
          'block5': < rede_oct4 >,
          'block6': < rede_oct4 >,
          'block7': < rede_oct4 >,
          'block8': < rede_oct4 >,
          'blocK': < bloco >,
          'mask1': < mascara_oct1 >,
          'mask2': < mascara_oct2 >,
          'mask3': < mascara_oct3 >,
          'mask4': < mascara_oct4 >,
          'mask5': < mascara_oct4 >,
          'mask6': < mascara_oct4 >,
          'mask7': < mascara_oct4 >,
          'mask8': < mascara_oct4 >,
          'active': < ativada >, }}

        :raise NetworkIPv6NotFoundError: NetworkIPV6 not found.
        :raise InvalidValueError: Invalid ID for NetworkIpv6
        :raise NetworkIPv6Error: Error in NetworkIpv6
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_network):
            raise InvalidParameterError(
                u'O id do rede ip6 foi informado incorretamente.')

        url = 'network/ipv6/id/' + str(id_network) + '/'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)