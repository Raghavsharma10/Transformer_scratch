def inserir(self, name):
        """Inserts a new Group L3 and returns its identifier.

        :param name: Group L3 name. String with a minimum 2 and maximum of 80 characters

        :return: Dictionary with the following structure:

        ::

            {'group_l3': {'id': < id_group_l3 >}}

        :raise InvalidParameterError: Name is null and invalid.
        :raise NomeGrupoL3DuplicadoError: There is already a registered Group L3 with the value of name.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        group_l3_map = dict()
        group_l3_map['name'] = name

        code, xml = self.submit({'group_l3': group_l3_map}, 'POST', 'groupl3/')

        return self.response(code, xml)