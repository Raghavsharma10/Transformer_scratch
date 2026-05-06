def remover(self, id_rack):
        """Remove Rack by the identifier.

        :param id_rack: Identifier of the Rack. Integer value and greater than zero.

        :return: None

        :raise InvalidParameterError: The identifier of Rack is null and invalid.
        :raise RackNaoExisteError: Rack not registered.
        :raise RackError: Rack is associated with a script.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        if not is_valid_int_param(id_rack):
            raise InvalidParameterError(
                u'The identifier of Rack is invalid or was not informed.')

        url = 'rack/' + str(id_rack) + '/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)