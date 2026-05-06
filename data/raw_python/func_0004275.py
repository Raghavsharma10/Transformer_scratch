def listar_por_tipo(self, id_script_type):
        """List all Script by Script Type.

        :param id_script_type: Identifier of the Script Type. Integer value and greater than zero.

        :return: Dictionary with the following structure:

        ::

            {‘script’: [{‘id’: < id >,
            ‘tipo_roteiro': < id_tipo_roteiro >,
            ‘nome': < nome >,
            ‘descricao’: < descricao >}, ...more Script...]}

        :raise InvalidParameterError: The identifier of Script Type is null and invalid.
        :raise TipoRoteiroNaoExisteError: Script Type not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        if not is_valid_int_param(id_script_type):
            raise InvalidParameterError(
                u'The identifier of Script Type is invalid or was not informed.')

        url = 'script/scripttype/' + str(id_script_type) + '/'

        code, map = self.submit(None, 'GET', url)

        key = 'script'
        return get_list_map(self.response(code, map, [key]), key)