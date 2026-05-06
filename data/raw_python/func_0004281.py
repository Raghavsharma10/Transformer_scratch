def get_network_ipv4(self, id_network):
        """
        Get networkipv4

        :param id_network: Identifier of the Network. Integer value and greater than zero.
        :return: Following dictionary:

        ::

          {'network': {'id': < id_networkIpv6 >,
          'network_type': < id_tipo_rede >,
          'ambiente_vip': < id_ambiente_vip >,
          'vlan': <id_vlan>
          'oct1': < rede_oct1 >,
          'oct2': < rede_oct2 >,
          'oct3': < rede_oct3 >,
          'oct4': < rede_oct4 >
          'blocK': < bloco >,
          'mask_oct1': < mascara_oct1 >,
          'mask_oct2': < mascara_oct2 >,
          'mask_oct3': < mascara_oct3 >,
          'mask_oct4': < mascara_oct4 >,
          'active': < ativada >,
          'broadcast':<'broadcast>, }}

        :raise NetworkIPv4NotFoundError: NetworkIPV4 not found.
        :raise InvalidValueError: Invalid ID for NetworkIpv4
        :raise NetworkIPv4Error: Error in NetworkIpv4
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_network):
            raise InvalidParameterError(
                u'O id do rede ip4 foi informado incorretamente.')

        url = 'network/ipv4/id/' + str(id_network) + '/'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)