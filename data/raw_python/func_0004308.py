def alter(self, id_option_vip, tipo_opcao, nome_opcao_txt):
        """Change Option VIP from by the identifier.

        :param id_option_vip: Identifier of the Option VIP. Integer value and greater than zero.
        :param tipo_opcao: Type. String with a maximum of 50 characters and respect [a-zA-Z\_-]
        :param nome_opcao_txt: Name Option. String with a maximum of 50 characters and respect [a-zA-Z\_-]

        :return: None

        :raise InvalidParameterError: Option VIP identifier is null and invalid.
        :raise InvalidParameterError: The value of tipo_opcao or nome_opcao_txt is invalid.
        :raise OptionVipNotFoundError: Option VIP not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_option_vip):
            raise InvalidParameterError(
                u'The identifier of Option VIP is invalid or was not informed.')

        optionvip_map = dict()
        optionvip_map['tipo_opcao'] = tipo_opcao
        optionvip_map['nome_opcao_txt'] = nome_opcao_txt

        url = 'optionvip/' + str(id_option_vip) + '/'

        code, xml = self.submit({'option_vip': optionvip_map}, 'PUT', url)

        return self.response(code, xml)