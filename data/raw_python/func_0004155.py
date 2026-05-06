def find_vlans(
            self,
            number,
            name,
            iexact,
            environment,
            net_type,
            network,
            ip_version,
            subnet,
            acl,
            pagination):
        """
        Find vlans by all search parameters

        :param number: Filter by vlan number column
        :param name: Filter by vlan name column
        :param iexact: Filter by name will be exact?
        :param environment: Filter by environment ID related
        :param net_type: Filter by network_type ID related
        :param network: Filter by each octs in network
        :param ip_version: Get only version (0:ipv4, 1:ipv6, 2:all)
        :param subnet: Filter by octs will search by subnets?
        :param acl: Filter by vlan acl column
        :param pagination: Class with all data needed to paginate

        :return: Following dictionary:

        ::

          {'vlan': {'id': < id_vlan >,
          'nome': < nome_vlan >,
          'num_vlan': < num_vlan >,
          'id_ambiente': < id_ambiente >,
          'descricao': < descricao >,
          'acl_file_name': < acl_file_name >,
          'acl_valida': < acl_valida >,
          'acl_file_name_v6': < acl_file_name_v6 >,
          'acl_valida_v6': < acl_valida_v6 >,
          'ativada': < ativada >,
          'ambiente_name': < divisao_dc-ambiente_logico-grupo_l3 >
          'redeipv4': [ { all networkipv4 related } ],
          'redeipv6': [ { all networkipv6 related } ] },
          'total': {< total_registros >} }

        :raise InvalidParameterError: Some parameter was invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not isinstance(pagination, Pagination):
            raise InvalidParameterError(
                u"Invalid parameter: pagination must be a class of type 'Pagination'.")

        vlan_map = dict()

        vlan_map['start_record'] = pagination.start_record
        vlan_map['end_record'] = pagination.end_record
        vlan_map['asorting_cols'] = pagination.asorting_cols
        vlan_map['searchable_columns'] = pagination.searchable_columns
        vlan_map['custom_search'] = pagination.custom_search

        vlan_map['numero'] = number
        vlan_map['nome'] = name
        vlan_map['exato'] = iexact
        vlan_map['ambiente'] = environment
        vlan_map['tipo_rede'] = net_type
        vlan_map['rede'] = network
        vlan_map['versao'] = ip_version
        vlan_map['subrede'] = subnet
        vlan_map['acl'] = acl

        url = 'vlan/find/'

        code, xml = self.submit({'vlan': vlan_map}, 'POST', url)

        key = 'vlan'
        return get_list_map(
            self.response(
                code, xml, [
                    key, 'redeipv4', 'redeipv6', 'equipamentos']), key)