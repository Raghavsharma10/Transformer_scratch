def listar_por_ambiente(self, id_ambiente):
        """List all VLANs from an environment.
        ** The itens returning from network is there to be compatible with other system **
        :param id_ambiente: Environment identifier.

        :return: Following dictionary:

        ::

          {'vlan': [{'id': < id_vlan >,
          'nome': < nome_vlan >,
          'num_vlan': < num_vlan >,
          'ambiente': < id_ambiente >,
          'descricao': < descricao >,
          'acl_file_name': < acl_file_name >,
          'acl_valida': < acl_valida >,
          'acl_file_name_v6': < acl_file_name_v6 >,
          'acl_valida_v6': < acl_valida_v6 >,
          'ativada': < ativada >,
          'id_tipo_rede': < id_tipo_rede >,
          'rede_oct1': < rede_oct1 >,
          'rede_oct2': < rede_oct2 >,
          'rede_oct3': < rede_oct3 >,
          'rede_oct4': < rede_oct4 >,
          'bloco': < bloco >,
          'mascara_oct1': < mascara_oct1 >,
          'mascara_oct2': < mascara_oct2 >,
          'mascara_oct3': < mascara_oct3 >,
          'mascara_oct4': < mascara_oct4 >,
          'broadcast': < broadcast >,} , ... other vlans ... ]}

        :raise InvalidParameterError: Environment id is none or invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        if not is_valid_int_param(id_ambiente):
            raise InvalidParameterError(u'Environment id is none or invalid.')

        url = 'vlan/ambiente/' + str(id_ambiente) + '/'

        code, xml = self.submit(None, 'GET', url)

        key = 'vlan'
        return get_list_map(self.response(code, xml, [key]), key)