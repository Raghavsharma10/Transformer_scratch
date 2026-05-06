def add_ipv6(self, id_network_ipv6, id_equip, description):
        """Allocate an IP on a network to an equipment.
        Insert new IP for network and associate to the equipment

        :param id_network_ipv6: ID for NetworkIPv6.
        :param id_equip: ID for Equipment.
        :param description: Description for IP.

        :return: Following dictionary:

        ::

            {'ip': {'id': < id_ip >,
            'id_network_ipv6': < id_network_ipv6 >,
            'bloco1': < bloco1 >,
            'bloco2': < bloco2 >,
            'bloco3': < bloco3 >,
            'bloco4': < bloco4 >,
            'bloco5': < bloco5 >,
            'bloco6': < bloco6 >,
            'bloco7': < bloco7 >,
            'bloco8': < bloco8 >,
            'descricao': < descricao >}}

        :raise InvalidParameterError: NetworkIPv6 identifier or Equipament identifier  is null and invalid,
        :raise InvalidParameterError: The value of description is invalid.
        :raise EquipamentoNaoExisteError: Equipment not found.
        :raise RedeIPv6NaoExisteError: NetworkIPv6 not found.
        :raise IPNaoDisponivelError: There is no network address is available to create the VLAN.
        :raise ConfigEnvironmentInvalidError: Invalid Environment Configuration or not registered
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        ip_map = dict()
        ip_map['id_network_ipv6'] = id_network_ipv6
        ip_map['description'] = description
        ip_map['id_equip'] = id_equip

        code, xml = self.submit({'ip': ip_map}, 'POST', 'ipv6/')

        return self.response(code, xml)