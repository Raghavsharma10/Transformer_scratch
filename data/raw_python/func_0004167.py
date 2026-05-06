def create_ipv6(self, id_network_ipv6):
        """Create VLAN in layer 2 using script 'navlan'.

        :param id_network_ipv6: NetworkIPv6 ID.

        :return: Following dictionary:

        ::

          {‘sucesso’: {‘codigo’: < codigo >,
          ‘descricao’: {'stdout':< stdout >, 'stderr':< stderr >}}}

        :raise NetworkIPv6NaoExisteError: NetworkIPv6 not found.
        :raise EquipamentoNaoExisteError: Equipament in list not found.
        :raise VlanError: VLAN is active.
        :raise InvalidParameterError: VLAN identifier is none or invalid.
        :raise InvalidParameterError: Equipment list is none or empty.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        :raise ScriptError: Failed to run the script.
        """

        url = 'vlan/v6/create/'

        vlan_map = dict()
        vlan_map['id_network_ip'] = id_network_ipv6

        code, xml = self.submit({'vlan': vlan_map}, 'POST', url)

        return self.response(code, xml)