def remover(self, id_brand):
        """Remove Brand from by the identifier.

        :param id_brand: Identifier of the Brand. Integer value and greater than zero.

        :return: None

        :raise InvalidParameterError: The identifier of Brand is null and invalid.
        :raise MarcaNaoExisteError: Brand not registered.
        :raise MarcaError: The brand is associated with a model.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_brand):
            raise InvalidParameterError(
                u'The identifier of Brand is invalid or was not informed.')

        url = 'brand/' + str(id_brand) + '/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)