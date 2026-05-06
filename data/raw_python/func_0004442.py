def get_all_option_pool(self, option_type=None):
        """Get all Option Pool.

        :return: Dictionary with the following structure:

        ::

            {[{‘id’: < id >,
            ‘type’: < tipo_opcao >,
            ‘name’: < nome_opcao_txt >}, ... other option pool ...] }

        :raise optionpoolNotFoundError: Option Pool not registered.
        :raise DataBaseError: Can't connect to networkapi database.
        :raise XMLError: Failed to generate the XML response.
        """
        if option_type:
            url = 'api/pools/options/?type='+option_type
        else:
            url = 'api/pools/options/'


        return self.get(url)