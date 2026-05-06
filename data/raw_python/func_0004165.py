def listar_permissao(self, nome_equipamento, nome_interface):
        """List all VLANS having communication permission to trunk from a port in switch.

        Run script 'configurador'.

        ::

          The value of 'stdout' key of return dictionary can have a list of numbers or
          number intervals of VLAN´s, comma separated. Examples of possible returns of 'stdout' below:
              - 100,103,111,...
              - 100-110,...
              - 100-110,112,115,...
              - 100,103,105-111,113,115-118,...

        :param nome_equipamento: Equipment name.
        :param nome_interface: Interface name.

        :return: Following dictionary:

        ::

          {‘sucesso’: {‘codigo’: < codigo >,
          ‘descricao’: {'stdout':< stdout >, 'stderr':< stderr >}}}

        :raise InvalidParameterError: Equipment name and/or interface name is invalid or none.
        :raise EquipamentoNaoExisteError: Equipment does not exist.
        :raise LigacaoFrontInterfaceNaoExisteError: There is no interface on front link of informed interface.
        :raise InterfaceNaoExisteError: Interface does not exist or is not associated to equipment.
        :raise LigacaoFrontNaoTerminaSwitchError: Interface does not have switch connected.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        :raise ScriptError: Failed to run the script.
        """
        vlan_map = dict()
        vlan_map['nome'] = nome_equipamento
        vlan_map['nome_interface'] = nome_interface

        code, xml = self.submit({'equipamento': vlan_map}, 'PUT', 'vlan/list/')

        return self.response(code, xml)