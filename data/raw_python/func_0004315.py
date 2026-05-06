def buscar_idtrafficreturn_opcvip(self, nome_opcao_txt):
        """Search id of Option VIP when tipo_opcao = 'Retorno de trafego' ​​

        :return: Dictionary with the following structure:

        ::

            {‘trafficreturn_opt’: ‘trafficreturn_opt’: <'id'>}

        :raise InvalidParameterError: Environment VIP identifier is null and invalid.
        :raise EnvironmentVipNotFoundError: Environment VIP not registered.
        :raise InvalidParameterError: finalidade_txt and cliente_txt is null and invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        trafficreturn_map = dict()
        trafficreturn_map['trafficreturn'] = nome_opcao_txt

        url = 'optionvip/trafficreturn/search/'

        code, xml = self.submit({'trafficreturn_opt':trafficreturn_map }, 'POST', url)

        return self.response(code, xml)