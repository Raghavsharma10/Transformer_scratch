def listar(self):
        """List all access types.

        :return: Dictionary with structure:

        ::

            {‘tipo_acesso’: [{‘id’: < id >,
            ‘protocolo’: < protocolo >}, ... other access types ...]}

        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        code, map = self.submit(None, 'GET', 'tipoacesso/')

        key = 'tipo_acesso'
        return get_list_map(self.response(code, map, [key]), key)