def remover(self, id_net_type):
        """Remove network type by its identifier.

        :param id_net_type: Network type identifier.

        :return: None

        :raise TipoRedeNaoExisteError: Network type does not exist.
        :raise TipoRedeError: Network type is associated with network.
        :raise InvalidParameterError: Network type is none or invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        if not is_valid_int_param(id_net_type):
            raise InvalidParameterError(
                u'Network type is invalid or was not informed.')

        url = 'net_type/' + str(id_net_type) + '/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)