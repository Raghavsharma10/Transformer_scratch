def get_real_related(self, id_equip):
        """
        Find reals related with equipment

        :param id_equip: Identifier of equipment

        :return: Following dictionary:

        ::

            {'vips': [{'port_real': < port_real >,
            'server_pool_member_id': < server_pool_member_id >,
            'ip': < ip >,
            'port_vip': < port_vip >,
            'host_name': < host_name >,
            'id_vip': < id_vip >, ...],
            'equip_name': < equip_name > }}

        :raise EquipamentoNaoExisteError: Equipment not registered.
        :raise InvalidParameterError: Some parameter was invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        url = 'equipamento/get_real_related/' + str(id_equip) + '/'

        code, xml = self.submit(None, 'GET', url)

        data = self.response(code, xml)
        return data