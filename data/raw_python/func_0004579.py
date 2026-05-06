def listar(self):
        """List all network types.

        :return: Following dictionary:

        ::

            {'net_type': [{'id': < id_tipo_rede >,
            'name': < nome_tipo_rede >}, ... other network types ...]}

        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        code, xml = self.submit(None, 'GET', 'net_type/')

        key = 'net_type'
        return get_list_map(self.response(code, xml, [key]), key)