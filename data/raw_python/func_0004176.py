def create_script_acl(self, id_vlan, network_type):
        '''Generate the script acl

        :param id_vlan: Vlan Id
        :param network_type: v4 or v6

        :raise InvalidValueError: Attrs invalids.
        :raise XMLError: Networkapi failed to generate the XML response.
        :raise VlanACLDuplicatedError: ACL name duplicate.
        :raise VlanNotFoundError: Vlan not registered.

        :return: Following dictionary:

        ::

          {'vlan': {
          'id': < id >,
          'nome': '< nome >',
          'num_vlan': < num_vlan >,
          'descricao': < descricao >
          'acl_file_name': < acl_file_name >,
          'ativada': < ativada >,
          'acl_valida': < acl_valida >,
          'acl_file_name_v6': < acl_file_name_v6 >,
          'redeipv6': < redeipv6 >,
          'acl_valida_v6': < acl_valida_v6 >,
          'redeipv4': < redeipv4 >,
          'ambiente': < ambiente >,
          }}
        '''

        vlan_map = dict()
        vlan_map['id_vlan'] = id_vlan
        vlan_map['network_type'] = network_type

        url = 'vlan/create/script/acl/'

        code, xml = self.submit({'vlan': vlan_map}, 'POST', url)

        return self.response(code, xml)