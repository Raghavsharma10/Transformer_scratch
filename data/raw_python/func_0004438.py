def add(self, tipo_opcao, nome_opcao):
        """Inserts a new Option Pool and returns its identifier.

        :param tipo_opcao: Type. String with a maximum of 50 characters and respect [a-zA-Z\_-]
        :param nome_opcao_txt: Name Option. String with a maximum of 50 characters and respect [a-zA-Z\_-]

        :return: Following dictionary:

        ::

            {'id': < id > , 'type':<type>, 'name':<name>}

        :raise InvalidParameterError: The value of tipo_opcao or nome_opcao_txt is invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        #optionpool_map = dict()
        #optionpool_map['type'] = tipo_opcao
        #optionpool_map['name'] = nome_opcao

        url='api/pools/options/save/'

        return self.post(url, {'type': tipo_opcao, "name":nome_opcao })