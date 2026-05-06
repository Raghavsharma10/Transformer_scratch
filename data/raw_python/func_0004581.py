def alterar(self, id_net_type, name):
        """Edit network type by its identifier.

        :param id_net_type: Network type identifier.
        :param name: Network type name.

        :return: None

        :raise InvalidParameterError: Network type identifier and/or name is(are) none or invalid.
        :raise TipoRedeNaoExisteError: Network type does not exist.
        :raise NomeTipoRedeDuplicadoError: Network type name already exists.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_net_type):
            raise InvalidParameterError(
                u'Network type is invalid or was not informed.')

        url = 'net_type/' + str(id_net_type) + '/'

        net_type_map = dict()
        net_type_map['name'] = name

        code, xml = self.submit({'net_type': net_type_map}, 'PUT', url)

        return self.response(code, xml)