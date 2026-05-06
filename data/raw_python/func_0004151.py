def inserir(self, name):
        """Inserts a new Division Dc and returns its identifier.

        :param name: Division Dc name. String with a minimum 2 and maximum of 80 characters

        :return: Dictionary with the following structure:

        ::

            {'division_dc': {'id': < id_division_dc >}}

        :raise InvalidParameterError: Name is null and invalid.
        :raise NomeDivisaoDcDuplicadoError: There is already a registered Division Dc with the value of name.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        division_dc_map = dict()
        division_dc_map['name'] = name

        code, xml = self.submit(
            {'division_dc': division_dc_map}, 'POST', 'divisiondc/')

        return self.response(code, xml)