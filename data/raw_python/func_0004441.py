def get_option_pool(self, id_option_pool):
        """Search Option Pool by id.

        :param id_option_pool: Identifier of the Option Pool. Integer value and greater than zero.

        :return: Following dictionary:

        ::

            {‘id’: < id_option_pool >,
            ‘type’: < tipo_opcao >,
            ‘name’: < nome_opcao_txt >}

        :raise InvalidParameterError: Option Pool identifier is null and invalid.
        :raise optionpoolNotFoundError: Option Pool not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_option_pool):
            raise InvalidParameterError(
                u'The identifier of Option Pool is invalid or was not informed.')

        url = 'api/pools/options/' + str(id_option_pool) + '/'

        return self.get(url)