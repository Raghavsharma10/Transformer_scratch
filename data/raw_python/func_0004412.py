def alterar(self, id_brand, name):
        """Change Brand from by the identifier.

        :param id_brand: Identifier of the Brand. Integer value and greater than zero.
        :param name: Brand name. String with a minimum 3 and maximum of 100 characters

        :return: None

        :raise InvalidParameterError: The identifier of Brand or name is null and invalid.
        :raise NomeMarcaDuplicadoError: There is already a registered Brand with the value of name.
        :raise MarcaNaoExisteError: Brand not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_brand):
            raise InvalidParameterError(
                u'The identifier of Brand is invalid or was not informed.')

        url = 'brand/' + str(id_brand) + '/'

        brand_map = dict()
        brand_map['name'] = name

        code, xml = self.submit({'brand': brand_map}, 'PUT', url)

        return self.response(code, xml)