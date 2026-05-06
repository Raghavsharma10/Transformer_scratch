def inserir(self, name):
        """Insert new network type and return its identifier.

        :param name: Network type name.

        :return: Following dictionary: {'net_type': {'id': < id >}}

        :raise InvalidParameterError: Network type is none or invalid.
        :raise NomeTipoRedeDuplicadoError: A network type with this name already exists.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        net_type_map = dict()
        net_type_map['name'] = name

        code, xml = self.submit(
            {'net_type': net_type_map}, 'POST', 'net_type/')

        return self.response(code, xml)