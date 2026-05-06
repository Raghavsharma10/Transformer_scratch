def get_option_vip(self, id_environment_vip):
        """Get all Option VIP by Environment Vip.

        :return: Dictionary with the following structure:

        ::

            {‘option_vip’: [{‘id’: < id >,
            ‘tipo_opcao’: < tipo_opcao >,
            ‘nome_opcao_txt’: < nome_opcao_txt >}, ... too option vips ...] }

        :raise EnvironmentVipNotFoundError: Environment VIP not registered.
        :raise DataBaseError: Can't connect to networkapi database.
        :raise XMLError: Failed to generate the XML response.
        """

        url = 'optionvip/environmentvip/' + str(id_environment_vip) + '/'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml, ['option_vip'])