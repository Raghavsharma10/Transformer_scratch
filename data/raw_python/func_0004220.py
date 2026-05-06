def remover(self, id_equipment, id_environment):
        """Remove Related Equipment with Environment from by the identifier.

        :param id_equipment: Identifier of the Equipment. Integer value and greater than zero.
        :param id_environment: Identifier of the Environment. Integer value and greater than zero.

        :return: None

        :raise InvalidParameterError:  The identifier of Environment, Equipament is null and invalid.
        :raise EquipamentoNotFoundError: Equipment not registered.
        :raise EquipamentoAmbienteNaoExisteError: Environment not registered.
        :raise VipIpError: IP-related equipment is being used for a request VIP.
        :raise XMLError: Networkapi failed to generate the XML response.
        :raise DataBaseError: Networkapi failed to access the database.
        """

        if not is_valid_int_param(id_equipment):
            raise InvalidParameterError(
                u'The identifier of Equipment is invalid or was not informed.')

        if not is_valid_int_param(id_environment):
            raise InvalidParameterError(
                u'The identifier of Environment is invalid or was not informed.')

        url = 'equipment/' + \
            str(id_equipment) + '/environment/' + str(id_environment) + '/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)