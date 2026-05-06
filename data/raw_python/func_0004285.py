def add_network_ipv6_hosts(
            self,
            id_vlan,
            id_tipo_rede,
            num_hosts,
            id_ambiente_vip=None):
        """
        Add new networkipv6

        :param id_vlan: Identifier of the Vlan. Integer value and greater than zero.
        :param id_tipo_rede: Identifier of the NetworkType. Integer value and greater than zero.
        :param num_hosts: Number of hosts expected. Integer value and greater than zero.
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
          'rede_oct5': < rede_oct4 >,
          'rede_oct6': < rede_oct4 >,
          'rede_oct7': < rede_oct4 >,
          'rede_oct8': < rede_oct4 >,
          'bloco': < bloco >,
          'mascara_oct1': < mascara_oct1 >,
          'mascara_oct2': < mascara_oct2 >,
          'mascara_oct3': < mascara_oct3 >,
          'mascara_oct4': < mascara_oct4 >,
          'mascara_oct5': < mascara_oct4 >,
          'mascara_oct6': < mascara_oct4 >,
          'mascara_oct7': < mascara_oct4 >,
          'mascara_oct8': < mascara_oct4 >,
          'broadcast': < broadcast >,
          'descricao': < descricao >,
          'acl_file_name': < acl_file_name >,
          'acl_valida': < acl_valida >,
          'ativada': < ativada >}}

        :raise TipoRedeNaoExisteError: NetworkType not found.
        :raise InvalidParameterError: Invalid ID for Vlan or NetworkType.
        :raise EnvironmentVipNotFoundError: Environment VIP not registered.
        :raise IPNaoDisponivelError: Network address unavailable to create a NetworkIPv6.
        :raise ConfigEnvironmentInvalidError: Invalid Environment Configuration or not registered
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        vlan_map = dict()
        vlan_map['id_vlan'] = id_vlan
        vlan_map['id_tipo_rede'] = id_tipo_rede
        vlan_map['num_hosts'] = num_hosts
        vlan_map['id_ambiente_vip'] = id_ambiente_vip

        code, xml = self.submit({'vlan': vlan_map}, 'PUT', 'network/ipv6/add/')

        return self.response(code, xml)