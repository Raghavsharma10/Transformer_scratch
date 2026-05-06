def alterar(self, id_model, id_brand, name):
        """Change Model from by the identifier.

        :param id_model: Identifier of the Model. Integer value and greater than zero.
        :param id_brand: Identifier of the Brand. Integer value and greater than zero.
        :param name: Model name. String with a minimum 3 and maximum of 100 characters

        :return: None

        :raise InvalidParameterError: The identifier of Model, Brand or name is null and invalid.
        :raise MarcaNaoExisteError: Brand not registered.
        :raise ModeloEquipamentoNaoExisteError: Model not registered.
        :raise NomeMarcaModeloDuplicadoError: There is already a registered Model with the value of name and brand.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response
        """

        if not is_valid_int_param(id_model):
            raise InvalidParameterError(
                u'The identifier of Model is invalid or was not informed.')

        model_map = dict()
        model_map['name'] = name
        model_map['id_brand'] = id_brand

        url = 'model/' + str(id_model) + '/'

        code, xml = self.submit({'model': model_map}, 'PUT', url)

        return self.response(code, xml)