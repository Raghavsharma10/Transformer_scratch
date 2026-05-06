def update(self, id_equipment, id_environment, is_router):
        """Remove Related Equipment with Environment from by the identifier.

        :param id_equipment: Identifier of the Equipment. Integer value and greater than zero.
        :param id_environment: Identifier of the Environment. Integer value and greater than zero.
        :param is_router: Identifier of the Environment. Boolean value.

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

        equipment_environment_map = dict()
        equipment_environment_map['id_equipamento'] = id_equipment
        equipment_environment_map['id_ambiente'] = id_environment
        equipment_environment_map['is_router'] = is_router

        code, xml = self.submit(
            {'equipamento_ambiente': equipment_environment_map}, 'PUT', 'equipamentoambiente/update/')

        return self.response(code, xml)