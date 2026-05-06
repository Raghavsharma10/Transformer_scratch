def insert_vlan(
            self,
            environment_id,
            name,
            number,
            description,
            acl_file,
            acl_file_v6,
            network_ipv4,
            network_ipv6,
            vrf=None):
        """Create new VLAN

        :param environment_id: ID for Environment.
        :param name: The name of VLAN.
        :param description: Some description to VLAN.
        :param number: Number of Vlan
        :param acl_file: Acl IPv4 File name to VLAN.
        :param acl_file_v6: Acl IPv6 File name to VLAN.
        :param network_ipv4: responsible for generating a network attribute ipv4 automatically.
        :param network_ipv6: responsible for generating a network attribute ipv6 automatically.

        :return: Following dictionary:

        ::

          {'vlan': {'id': < id_vlan >,
          'nome': < nome_vlan >,
          'num_vlan': < num_vlan >,
          'id_ambiente': < id_ambiente >,
          'descricao': < descricao >,
          'acl_file_name': < acl_file_name >,
          'acl_valida': < acl_valida >,
          'ativada': < ativada >
          'acl_file_name_v6': < acl_file_name_v6 >,
          'acl_valida_v6': < acl_valida_v6 >, } }

        :raise VlanError: VLAN name already exists, VLAN name already exists, DC division of the environment invalid or does not exist VLAN number available.
        :raise VlanNaoExisteError: VLAN not found.
        :raise AmbienteNaoExisteError: Environment not registered.
        :raise InvalidParameterError: Name of Vlan and/or the identifier of the Environment is null or invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(environment_id):
            raise InvalidParameterError(u'Environment id is none or invalid.')

        if not is_valid_int_param(number):
            raise InvalidParameterError(u'Vlan number is none or invalid')

        vlan_map = dict()
        vlan_map['environment_id'] = environment_id
        vlan_map['name'] = name
        vlan_map['description'] = description
        vlan_map['acl_file'] = acl_file
        vlan_map['acl_file_v6'] = acl_file_v6
        vlan_map['number'] = number
        vlan_map['network_ipv4'] = network_ipv4
        vlan_map['network_ipv6'] = network_ipv6
        vlan_map['vrf'] = vrf

        code, xml = self.submit({'vlan': vlan_map}, 'POST', 'vlan/insert/')

        return self.response(code, xml)