def remover(self, id_groupl3):
        """Remove Group L3 from by the identifier.

        :param id_groupl3: Identifier of the Group L3. Integer value and greater than zero.

        :return: None

        :raise InvalidParameterError: The identifier of Group L3 is null and invalid.
        :raise GrupoL3NaoExisteError: Group L3 not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_groupl3):
            raise InvalidParameterError(
                u'The identifier of Group L3 is invalid or was not informed.')

        url = 'groupl3/' + str(id_groupl3) + '/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)