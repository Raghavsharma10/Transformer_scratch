def search(self, id_egroup):
        """Search Group Equipament from by the identifier.

        :param id_egroup: Identifier of the Group Equipament. Integer value and greater than zero.

        :return: Following dictionary:

        ::

            {‘group_equipament’:  {‘id’: < id_egrupo >,
            ‘nome’: < nome >} }

        :raise InvalidParameterError: Group Equipament identifier is null and invalid.
        :raise GrupoEquipamentoNaoExisteError: Group Equipament not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_egroup):
            raise InvalidParameterError(
                u'The identifier of Group Equipament is invalid or was not informed.')

        url = 'egroup/' + str(id_egroup) + '/'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)