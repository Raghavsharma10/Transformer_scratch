def create_ipv4(self, id_network_ipv4):
        """Create VLAN in layer 2 using script 'navlan'.

        :param id_network_ipv4: NetworkIPv4 ID.

        :return: Following dictionary:

        ::

          {‘sucesso’: {‘codigo’: < codigo >,
          ‘descricao’: {'stdout':< stdout >, 'stderr':< stderr >}}}

        :raise NetworkIPv4NaoExisteError: NetworkIPv4 not found.
        :raise EquipamentoNaoExisteError: Equipament in list not found.
        :raise VlanError: VLAN is active.
        :raise InvalidParameterError: VLAN identifier is none or invalid.
        :raise InvalidParameterError: Equipment list is none or empty.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        :raise ScriptError: Failed to run the script.
        """

        url = 'vlan/v4/create/'

        vlan_map = dict()
        vlan_map['id_network_ip'] = id_network_ipv4

        code, xml = self.submit({'vlan': vlan_map}, 'POST', url)

        return self.response(code, xml)