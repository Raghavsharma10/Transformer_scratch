def get(self, id_vlan):
        """Get a VLAN by your primary key.
        Network IPv4/IPv6 related will also be fetched.

        :param id_vlan: ID for VLAN.

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
          'redeipv4': [ { all networkipv4 related } ],
          'redeipv6': [ { all networkipv6 related } ] } }

        :raise InvalidParameterError: Invalid ID for VLAN.
        :raise VlanNaoExisteError: VLAN not found.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        if not is_valid_int_param(id_vlan):
            raise InvalidParameterError(
                u'Parameter id_vlan is invalid. Value: ' +
                id_vlan)

        url = 'vlan/' + str(id_vlan) + '/network/'

        code, xml = self.submit(None, 'GET', url)

        return get_list_map(
            self.response(
                code, xml, [
                    'redeipv4', 'redeipv6']), 'vlan')