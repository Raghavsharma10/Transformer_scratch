def buscar_ambientep44_por_finalidade_cliente(
            self,
            finalidade_txt,
            cliente_txt):
        """Search ambiente_p44_txt environment vip

        :return: Dictionary with the following structure:

        ::

            {‘ambiente_p44_txt’:
            'id':<'id_ambientevip'>,
            ‘finalidade’: <'finalidade_txt'>,
            'cliente_txt: <'cliente_txt'>',
            'ambiente_p44: <'ambiente_p44'>',}

        :raise InvalidParameterError: finalidade_txt and cliente_txt is null and invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        vip_map = dict()
        vip_map['finalidade_txt'] = finalidade_txt
        vip_map['cliente_txt'] = cliente_txt

        url = 'environment-vip/get/ambiente_p44_txt/'

        code, xml = self.submit({'vip': vip_map}, 'POST', url)

        return self.response(code, xml)