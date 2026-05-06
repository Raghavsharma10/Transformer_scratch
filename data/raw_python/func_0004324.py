def remover(self, id_interface):
        """Remove an interface by its identifier.

        :param id_interface: Interface identifier.

        :return: None

        :raise InterfaceNaoExisteError: Interface doesn't exist.
        :raise InterfaceError: Interface is linked to another interface.
        :raise InvalidParameterError: The interface identifier is invalid or none.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        if not is_valid_int_param(id_interface):
            raise InvalidParameterError(
                u'Interface id is invalid or was not informed.')

        url = 'interface/' + str(id_interface) + '/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)