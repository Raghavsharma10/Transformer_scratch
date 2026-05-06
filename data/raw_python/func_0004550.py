def buscar_cliente_por_finalidade(self, finalidade_txt):
        """Search cliente_txt environment vip

        :return: Dictionary with the following structure:

        ::

            {‘cliente_txt’:
            ‘finalidade’: <'finalidade_txt'>,
            'cliente_txt: <'cliente_txt'>'}

        :raise InvalidParameterError: finalidade_txt is null and invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        vip_map = dict()
        vip_map['finalidade_txt'] = finalidade_txt

        url = 'environment-vip/get/cliente_txt/'

        code, xml = self.submit({'vip': vip_map}, 'POST', url)

        return self.response(code, xml)