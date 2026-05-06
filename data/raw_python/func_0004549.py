def buscar_finalidade(self):
        """Search finalidade_txt environment vip

        :return: Dictionary with the following structure:

        ::
            {‘finalidade’:  ‘finalidade’: <'finalidade_txt'>}

        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        url = 'environment-vip/get/finality'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)