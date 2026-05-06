def allocate_IPv6(
            self,
            name,
            id_network_type,
            id_environment,
            description,
            id_environment_vip=None):
        """Inserts a new VLAN.

        :param name: Name of Vlan. String with a maximum of 50 characters.
        :param id_network_type: Identifier of the Netwok Type. Integer value and greater than zero.
        :param id_environment: Identifier of the Environment. Integer value and greater than zero.
        :param description: Description of Vlan. String with a maximum of 200 characters.
        :param id_environment_vip: Identifier of the Environment Vip. Integer value and greater than zero.

        :return: Following dictionary:

        ::

          {'vlan': {'id': < id_vlan >,
          'nome': < nome_vlan >,
          'num_vlan': < num_vlan >,
          'id_tipo_rede': < id_tipo_rede >,
          'id_ambiente': < id_ambiente >,
          'bloco1': < bloco1 >,
          'bloco2': < bloco2 >,
          'bloco3': < bloco3 >,
          'bloco4': < bloco4 >,
          'bloco5': < bloco5 >,
          'bloco6': < bloco6 >,
          'bloco7': < bloco7 >,
          'bloco8': < bloco8 >,
          'bloco': < bloco >,
          'mask_bloco1': < mask_bloco1 >,
          'mask_bloco2': < mask_bloco2 >,
          'mask_bloco3': < mask_bloco3 >,
          'mask_bloco4': < mask_bloco4 >,
          'mask_bloco5': < mask_bloco5 >,
          'mask_bloco6': < mask_bloco6 >,
          'mask_bloco7': < mask_bloco7 >,
          'mask_bloco8': < mask_bloco8 >,
          'descricao': < descricao >,
          'acl_file_name': < acl_file_name >,
          'acl_valida': < acl_valida >,
          'acl_file_name_v6': < acl_file_name_v6 >,
          'acl_valida_v6': < acl_valida_v6 >,
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
        vlan_map['name'] = name
        vlan_map['id_network_type'] = id_network_type
        vlan_map['id_environment'] = id_environment
        vlan_map['description'] = description
        vlan_map['id_environment_vip'] = id_environment_vip

        code, xml = self.submit({'vlan': vlan_map}, 'POST', 'vlan/ipv6/')

        return self.response(code, xml)