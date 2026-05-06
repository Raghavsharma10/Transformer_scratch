def listar_por_marca(self, id_brand):
        """List all Model by Brand.

        :param id_brand: Identifier of the Brand. Integer value and greater than zero.

        :return: Dictionary with the following structure:

        ::

            {‘model’: [{‘id’: < id >,
            ‘nome’: < nome >,
            ‘id_marca’: < id_marca >}, ... too Model ...]}

        :raise InvalidParameterError: The identifier of Brand is null and invalid.
        :raise MarcaNaoExisteError: Brand not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response
        """

        if not is_valid_int_param(id_brand):
            raise InvalidParameterError(
                u'The identifier of Brand is invalid or was not informed.')

        url = 'model/brand/' + str(id_brand) + '/'

        code, map = self.submit(None, 'GET', url)

        key = 'model'
        return get_list_map(self.response(code, map, [key]), key)