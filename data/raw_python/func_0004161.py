def allocate_without_network(self, environment_id, name, description, vrf=None):
        """Create new VLAN without add NetworkIPv4.

        :param environment_id: ID for Environment.
        :param name: The name of VLAN.
        :param description: Some description to VLAN.

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
          'ativada': < ativada > } }

        :raise VlanError: Duplicate name of VLAN, division DC of Environment not found/invalid or VLAN number not available.
        :raise AmbienteNaoExisteError: Environment not found.
        :raise InvalidParameterError: Some parameter was invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        vlan_map = dict()
        vlan_map['environment_id'] = environment_id
        vlan_map['name'] = name
        vlan_map['description'] = description
        vlan_map['vrf'] = vrf

        code, xml = self.submit({'vlan': vlan_map}, 'POST', 'vlan/no-network/')

        return self.response(code, xml)