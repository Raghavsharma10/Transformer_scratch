def buscar(self, id_vlan):
        """Get VLAN by its identifier.

        :param id_vlan: VLAN identifier.

        :return: Following dictionary:

        ::

          {'vlan': {'id': < id_vlan >,
          'nome': < nome_vlan >,
          'num_vlan': < num_vlan >,
          'id_ambiente': < id_ambiente >,
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
          'broadcast': < broadcast >,
          'descricao': < descricao >,
          'acl_file_name': < acl_file_name >,
          'acl_valida': < acl_valida >,
          'ativada': < ativada >}
          OR {'id': < id_vlan >,
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
          'broadcast': < broadcast >,
          'descricao': < descricao >,
          'acl_file_name': < acl_file_name >,
          'acl_valida': < acl_valida >,
          'acl_file_name_v6': < acl_file_name_v6 >,
          'acl_valida_v6': < acl_valida_v6 >,
          'ativada': < ativada >}}

        :raise VlanNaoExisteError: VLAN does not exist.
        :raise InvalidParameterError: VLAN id is none or invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        if not is_valid_int_param(id_vlan):
            raise InvalidParameterError(
                u'Vlan id is invalid or was not informed.')

        url = 'vlan/' + str(id_vlan) + '/'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)