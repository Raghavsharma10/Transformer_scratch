def list_all(self):
        """
        List all environment vips

        :return: Following dictionary:

        ::

            {'environment_vip': [{'id': <id>,
            'finalidade_txt': <finalidade_txt>,
            'cliente_txt': <cliente_txt>,
            'ambiente_p44_txt': <ambiente_p44_txt> } {... other environments vip ...}]}

        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        url = 'environmentvip/all/'

        code, xml = self.submit(None, 'GET', url)

        key = 'environment_vip'

        return get_list_map(self.response(code, xml, [key]), key)