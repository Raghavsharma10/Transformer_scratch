def list_by_group(self, id_egroup):
        """Search Group Equipment from by the identifier.

        :param id_egroup: Identifier of the Group Equipment. Integer value and greater than zero.

        :return: Dictionary with the following structure:

        ::

            {'equipaments':
            [{'nome': < name_equipament >, 'grupos': < id_group >,
            'mark': {'id': < id_mark >, 'nome': < name_mark >},'modelo': < id_model >,
            'tipo_equipamento': < id_type >,
            'model': {'nome': , 'id': < id_model >, 'marca': < id_mark >},
            'type': {id': < id_type >, 'tipo_equipamento': < name_type >},
            'id': < id_equipment >}, ... ]}

        :raise InvalidParameterError: Group Equipment is null and invalid.
        :raise GrupoEquipamentoNaoExisteError: Group Equipment not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if id_egroup is None:
            raise InvalidParameterError(
                u'The identifier of Group Equipament is invalid or was not informed.')

        url = 'equipment/group/' + str(id_egroup) + '/'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)