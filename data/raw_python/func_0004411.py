def inserir(self, name):
        """Inserts a new Brand and returns its identifier

        :param name: Brand name. String with a minimum 3 and maximum of 100 characters

        :return: Dictionary with the following structure:

        ::

            {'marca': {'id': < id_brand >}}

        :raise InvalidParameterError: Name is null and invalid.
        :raise NomeMarcaDuplicadoError: There is already a registered Brand with the value of name.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        brand_map = dict()
        brand_map['name'] = name

        code, xml = self.submit({'brand': brand_map}, 'POST', 'brand/')

        return self.response(code, xml)