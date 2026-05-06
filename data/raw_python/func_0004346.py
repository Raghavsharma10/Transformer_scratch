def remover(self, id_script_type):
        """Remove Script Type from by the identifier.

        :param id_script_type: Identifier of the Script Type. Integer value and greater than zero.

        :return: None

        :raise InvalidParameterError: The identifier of Script Type is null and invalid.
        :raise TipoRoteiroNaoExisteError: Script Type not registered.
        :raise TipoRoteiroError: Script type is associated with a script.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        if not is_valid_int_param(id_script_type):
            raise InvalidParameterError(
                u'The identifier of Script Type is invalid or was not informed.')

        url = 'scripttype/' + str(id_script_type) + '/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)