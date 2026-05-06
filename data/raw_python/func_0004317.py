def inserir(self, id_brand, name):
        """Inserts a new Model and returns its identifier

        :param id_brand: Identifier of the Brand. Integer value and greater than zero.
        :param name: Model name. String with a minimum 3 and maximum of 100 characters

        :return: Dictionary with the following structure:

        ::

            {'model': {'id': < id_model >}}

        :raise InvalidParameterError: The identifier of Brand or name is null and invalid.
        :raise NomeMarcaModeloDuplicadoError: There is already a registered Model with the value of name and brand.
        :raise MarcaNaoExisteError: Brand not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response
        """
        model_map = dict()
        model_map['name'] = name
        model_map['id_brand'] = id_brand

        code, xml = self.submit({'model': model_map}, 'POST', 'model/')

        return self.response(code, xml)