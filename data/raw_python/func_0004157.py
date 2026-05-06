def alocar(
            self,
            nome,
            id_tipo_rede,
            id_ambiente,
            descricao,
            id_ambiente_vip=None,
            vrf=None):
        """Inserts a new VLAN.

        :param nome: Name of Vlan. String with a maximum of 50 characters.
        :param id_tipo_rede: Identifier of the Network Type. Integer value and greater than zero.
        :param id_ambiente: Identifier of the Environment. Integer value and greater than zero.
        :param descricao: Description of Vlan. String with a maximum of 200 characters.
        :param id_ambiente_vip: Identifier of the Environment Vip. Integer value and greater than zero.

        :return: Following dictionary:

        ::

          {'vlan': {'id': < id_vlan >,
          'nome': < nome_vlan >,
          'num_vlan': < num_vlan >,
          'id_tipo_rede': < id_tipo_rede >,
          'id_ambiente': < id_ambiente >,
          'rede_oct1': < rede_oct1 >,
          'rede_oct2': < rede_oct2 >,
          'rede_oct3': < rede_oct3 >,
          'rede_oct4': < rede_oct4 >,
          'bloco': < bloco >,
          'mascara_oct1': < mascara_oct1 >,
          'mascara_oct2': < mascara_oct2 >,
          'mascara_oct3': < mascara_oct3 >,
          'mascara_oct4': < mascara_oct4 >,
          'broadcast': < broadcast >,
          'descricao': < descricao >,
          'acl_file_name': < acl_file_name >,
          'acl_valida': < acl_valida >,
          'ativada': < ativada >}}

        :raise VlanError: VLAN name already exists, VLAN name already exists, DC division of the environment invalid or does not exist VLAN number available.
        :raise VlanNaoExisteError: VLAN not found.
        :raise TipoRedeNaoExisteError: Network Type not registered.
        :raise AmbienteNaoExisteError: Environment not registered.
        :raise EnvironmentVipNotFoundError: Environment VIP not registered.
        :raise InvalidParameterError: Name of Vlan and/or the identifier of the Environment is null or invalid.
        :raise IPNaoDisponivelError: There is no network address is available to create the VLAN.
        :raise ConfigEnvironmentInvalidError: Invalid Environment Configuration or not registered
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        vlan_map = dict()
        vlan_map['nome'] = nome
        vlan_map['id_tipo_rede'] = id_tipo_rede
        vlan_map['id_ambiente'] = id_ambiente
        vlan_map['descricao'] = descricao
        vlan_map['id_ambiente_vip'] = id_ambiente_vip
        vlan_map['vrf'] = vrf

        code, xml = self.submit({'vlan': vlan_map}, 'POST', 'vlan/')

        return self.response(code, xml)