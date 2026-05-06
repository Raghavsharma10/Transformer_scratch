def add_ipv4(self, id_network_ipv4, id_equipamento, descricao):
        """Allocate an IP on a network to an equipment.
        Insert new IP for network and associate to the equipment

        :param id_network_ipv4: ID for NetworkIPv4.
        :param id_equipamento: ID for Equipment.
        :param descricao: Description for IP.

        :return: Following dictionary:

        ::

            {'ip': {'id': < id_ip >,
            'id_network_ipv4': < id_network_ipv4 >,
            'oct1’: < oct1 >,
            'oct2': < oct2 >,
            'oct3': < oct3 >,
            'oct4': < oct4 >,
            'descricao': < descricao >}}

        :raise InvalidParameterError: Invalid ID for NetworkIPv4 or Equipment.
        :raise InvalidParameterError: The value of description is invalid.
        :raise EquipamentoNaoExisteError: Equipment not found.
        :raise RedeIPv4NaoExisteError: NetworkIPv4 not found.
        :raise IPNaoDisponivelError: There is no network address is available to create the VLAN.
        :raise ConfigEnvironmentInvalidError: Invalid Environment Configuration or not registered
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        ip_map = dict()
        ip_map['id_network_ipv4'] = id_network_ipv4
        ip_map['description'] = descricao
        ip_map['id_equipment'] = id_equipamento

        code, xml = self.submit({'ip': ip_map}, 'POST', 'ipv4/')

        return self.response(code, xml)