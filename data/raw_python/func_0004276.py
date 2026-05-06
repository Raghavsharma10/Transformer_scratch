def listar_por_equipamento(self, id_equipment):
        """List all Script related Equipment.

        :param id_equipment: Identifier of the Equipment. Integer value and greater than zero.

        :return: Dictionary with the following structure:

        ::

            {script': [{‘id’: < id >,
            ‘nome’: < nome >,
            ‘descricao’: < descricao >,
            ‘id_tipo_roteiro’: < id_tipo_roteiro >,
            ‘nome_tipo_roteiro’: < nome_tipo_roteiro >,
            ‘descricao_tipo_roteiro’: < descricao_tipo_roteiro >}, ...more Script...]}

        :raise InvalidParameterError: The identifier of Equipment is null and invalid.
        :raise EquipamentoNaoExisteError: Equipment not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        if not is_valid_int_param(id_equipment):
            raise InvalidParameterError(
                u'The identifier of Equipment is invalid or was not informed.')

        url = 'script/equipment/' + str(id_equipment) + '/'

        code, map = self.submit(None, 'GET', url)

        key = 'script'
        return get_list_map(self.response(code, map, [key]), key)