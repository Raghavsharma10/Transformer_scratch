def alterar(self, id_groupl3, name):
        """Change Group L3 from by the identifier.

        :param id_groupl3: Identifier of the Group L3. Integer value and greater than zero.
        :param name: Group L3 name. String with a minimum 2 and maximum of 80 characters

        :return: None

        :raise InvalidParameterError: The identifier of Group L3 or name is null and invalid.
        :raise NomeGrupoL3DuplicadoError: There is already a registered Group L3 with the value of name.
        :raise GrupoL3NaoExisteError: Group L3 not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_groupl3):
            raise InvalidParameterError(
                u'The identifier of Group L3 is invalid or was not informed.')

        url = 'groupl3/' + str(id_groupl3) + '/'

        group_l3_map = dict()
        group_l3_map['name'] = name

        code, xml = self.submit({'groupl3': group_l3_map}, 'PUT', url)

        return self.response(code, xml)