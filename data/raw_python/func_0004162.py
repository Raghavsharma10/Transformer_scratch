def verificar_permissao(self, id_vlan, nome_equipamento, nome_interface):
        """Check if there is communication permission for VLAN to trunk.

        Run script 'configurador'.

        The "stdout" key value of response dictionary is 1(one) if VLAN has permission,
        or 0(zero), otherwise.

        :param id_vlan: VLAN identifier.
        :param nome_equipamento: Equipment name.
        :param nome_interface: Interface name.

        :return: Following dictionary:

        ::

          {‘sucesso’: {‘codigo’: < codigo >,
          ‘descricao’: {'stdout':< stdout >, 'stderr':< stderr >}}}

        :raise VlanNaoExisteError: VLAN does not exist.
        :raise InvalidParameterError: VLAN id is none or invalid.
        :raise InvalidParameterError: Equipment name and/or interface name is invalid or none.
        :raise EquipamentoNaoExisteError: Equipment does not exist.
        :raise LigacaoFrontInterfaceNaoExisteError: There is no interface on front link of informed interface.
        :raise InterfaceNaoExisteError: Interface does not exist or is not associated to equipment.
        :raise LigacaoFrontNaoTerminaSwitchError: Interface does not have switch connected.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        :raise ScriptError: Failed to run the script.
        """
        if not is_valid_int_param(id_vlan):
            raise InvalidParameterError(
                u'Vlan id is invalid or was not informed.')

        url = 'vlan/' + str(id_vlan) + '/check/'

        vlan_map = dict()
        vlan_map['nome'] = nome_equipamento
        vlan_map['nome_interface'] = nome_interface

        code, xml = self.submit({'equipamento': vlan_map}, 'PUT', url)

        return self.response(code, xml)