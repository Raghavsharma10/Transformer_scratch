def remover(self, id_script):
        """Remove Script from by the identifier.

        :param id_script: Identifier of the Script. Integer value and greater than zero.

        :return: None

        :raise InvalidParameterError: The identifier of Script is null and invalid.
        :raise RoteiroNaoExisteError: Script not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        if not is_valid_int_param(id_script):
            raise InvalidParameterError(
                u'The identifier of Script is invalid or was not informed.')

        url = 'script/' + str(id_script) + '/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)