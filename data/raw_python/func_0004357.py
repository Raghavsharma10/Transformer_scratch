def edit(self, id_equip, nome, id_tipo_equipamento, id_modelo, maintenance=None):
        """Change Equipment from by the identifier.

        :param id_equip: Identifier of the Equipment. Integer value and greater than zero.
        :param nome: Equipment name. String with a minimum 3 and maximum of 30 characters
        :param id_tipo_equipamento: Identifier of the Equipment Type. Integer value and greater than zero.
        :param id_modelo: Identifier of the Model. Integer value and greater than zero.

        :return: None

        :raise InvalidParameterError: The identifier of Equipment, model, equipment type or name  is null and invalid.
        :raise EquipamentoNaoExisteError: Equipment not registered.
        :raise TipoEquipamentoNaoExisteError: Equipment Type not registered.
        :raise ModeloEquipamentoNaoExisteError: Model not registered.
        :raise GrupoEquipamentoNaoExisteError: Group not registered.

        :raise EquipamentoError: Equipamento com o nome duplicado ou
            Equipamento do grupo “Equipamentos Orquestração” somente poderá ser
            criado com tipo igual a “Servidor Virtual".

        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response

        """

        equip_map = dict()
        equip_map['id_equip'] = id_equip
        equip_map['id_tipo_equipamento'] = id_tipo_equipamento
        equip_map['id_modelo'] = id_modelo
        equip_map['nome'] = nome
        if maintenance is not None:
            equip_map['maintenance'] = maintenance

        url = 'equipamento/edit/' + str(id_equip) + '/'

        code, xml = self.submit({'equipamento': equip_map}, 'POST', url)

        return self.response(code, xml)