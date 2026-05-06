def alterar(self, id_script_type, type, description):
        """Change Script Type from by the identifier.

        :param id_script_type: Identifier of the Script Type. Integer value and greater than zero.
        :param type: Script Type type. String with a minimum 3 and maximum of 40 characters
        :param description: Script Type description. String with a minimum 3 and maximum of 100 characters

        :return: None

        :raise InvalidParameterError: The identifier of Script Type, type or description is null and invalid.
        :raise TipoRoteiroNaoExisteError: Script Type not registered.
        :raise NomeTipoRoteiroDuplicadoError: Type script already registered with informed.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        if not is_valid_int_param(id_script_type):
            raise InvalidParameterError(
                u'The identifier of Script Type is invalid or was not informed.')

        script_type_map = dict()
        script_type_map['type'] = type
        script_type_map['description'] = description

        url = 'scripttype/' + str(id_script_type) + '/'

        code, xml = self.submit({'script_type': script_type_map}, 'PUT', url)

        return self.response(code, xml)