def remover(self, id_model):
        """Remove Model from by the identifier.

        :param id_model: Identifier of the Model. Integer value and greater than zero.

        :return: None

        :raise InvalidParameterError: The identifier of Model is null and invalid.
        :raise ModeloEquipamentoNaoExisteError: Model not registered.
        :raise ModeloEquipamentoError: The Model is associated with a equipment.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response
        """

        if not is_valid_int_param(id_model):
            raise InvalidParameterError(
                u'The identifier of Model is invalid or was not informed.')

        url = 'model/' + str(id_model) + '/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)