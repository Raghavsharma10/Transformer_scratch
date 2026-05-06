def modify(self, id_option_pool, tipo_opcao, nome_opcao):
        """Change Option Pool from by id.

        :param id_option_pool: Identifier of the Option Pool. Integer value and greater than zero.
        :param tipo_opcao: Type. String with a maximum of 50 characters and respect [a-zA-Z\_-]
        :param nome_opcao_txt: Name Option. String with a maximum of 50 characters and respect [a-zA-Z\_-]

        :return: None

        :raise InvalidParameterError: Option Pool identifier is null or invalid.
        :raise InvalidParameterError: The value of tipo_opcao or nome_opcao_txt is invalid.
        :raise optionpoolNotFoundError: Option pool not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_option_pool):
            raise InvalidParameterError(
                u'The identifier of Option Pool is invalid or was not informed.')

        #optionpool_map = dict()
        #optionpool_map['type'] = tipo_opcao
        #optionpool_map['name'] = nome_opcao_txt

        url = 'api/pools/options/' + str(id_option_pool) + '/'

        return self.put(url,{'type': tipo_opcao, "name":nome_opcao } )