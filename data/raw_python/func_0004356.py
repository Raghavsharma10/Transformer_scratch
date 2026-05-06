def inserir(self, name, id_equipment_type, id_model, id_group, maintenance=False):
        """Inserts a new Equipment and returns its identifier

        Além de inserir o equipamento, a networkAPI também associa o equipamento
        ao grupo informado.

        :param name: Equipment name. String with a minimum 3 and maximum of 30 characters
        :param id_equipment_type: Identifier of the Equipment Type. Integer value and greater than zero.
        :param id_model: Identifier of the Model. Integer value and greater than zero.
        :param id_group: Identifier of the Group. Integer value and greater than zero.

        :return: Dictionary with the following structure:

        ::

            {'equipamento': {'id': < id_equipamento >},
            'equipamento_grupo': {'id': < id_grupo_equipamento >}}

        :raise InvalidParameterError: The identifier of Equipment type, model, group or name  is null and invalid.
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
        equip_map['id_tipo_equipamento'] = id_equipment_type
        equip_map['id_modelo'] = id_model
        equip_map['nome'] = name
        equip_map['id_grupo'] = id_group
        equip_map['maintenance'] = maintenance
        
        code, xml = self.submit(
            {'equipamento': equip_map}, 'POST', 'equipamento/')

        return self.response(code, xml)