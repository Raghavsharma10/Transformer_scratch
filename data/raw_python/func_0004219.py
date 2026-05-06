def inserir(self, id_equipment, id_environment, is_router=0):
        """Inserts a new Related Equipment with Environment and returns its identifier

        :param id_equipment: Identifier of the Equipment. Integer value and greater than zero.
        :param id_environment: Identifier of the Environment. Integer value and greater than zero.
        :param is_router: Identifier of the Environment. Boolean value.

        :return: Dictionary with the following structure:

        ::

            {'equipamento_ambiente': {'id': < id_equipment_environment >}}

        :raise InvalidParameterError: The identifier of Equipment or Environment is null and invalid.
        :raise AmbienteNaoExisteError: Environment not registered.
        :raise EquipamentoNaoExisteError: Equipment not registered.
        :raise EquipamentoAmbienteError: Equipment is already associated with the Environment.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        equipment_environment_map = dict()
        equipment_environment_map['id_equipamento'] = id_equipment
        equipment_environment_map['id_ambiente'] = id_environment
        equipment_environment_map['is_router'] = is_router

        code, xml = self.submit(
            {'equipamento_ambiente': equipment_environment_map}, 'POST', 'equipamentoambiente/')

        return self.response(code, xml)