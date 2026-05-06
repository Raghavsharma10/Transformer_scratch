def add(self, tipo_opcao, nome_opcao_txt):
        """Inserts a new Option VIP and returns its identifier.

        :param tipo_opcao: Type. String with a maximum of 50 characters and respect [a-zA-Z\_-]
        :param nome_opcao_txt: Name Option. String with a maximum of 50 characters and respect [a-zA-Z\_-]

        :return: Following dictionary:

        ::

            {'option_vip': {'id': < id >}}

        :raise InvalidParameterError: The value of tipo_opcao or nome_opcao_txt is invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        optionvip_map = dict()
        optionvip_map['tipo_opcao'] = tipo_opcao
        optionvip_map['nome_opcao_txt'] = nome_opcao_txt

        code, xml = self.submit(
            {'option_vip': optionvip_map}, 'POST', 'optionvip/')

        return self.response(code, xml)