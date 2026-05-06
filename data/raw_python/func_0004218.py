def inserir(self, name):
        """Inserts a new Equipment Type and returns its identifier

        :param name: Equipment Type name. String with a minimum 3 and maximum of 100 characters

        :return: Dictionary with the following structure:

        ::

            {'tipo_equipamento': {'id': < id_equipment_ype >}}

        :raise InvalidParameterError: Name is null and invalid.
        :raise EquipamentoError: There is already a registered Equipment Type with the value of name.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response
        """
        equipment_type_map = dict()
        equipment_type_map['name'] = name

        url = 'equipmenttype/'

        code, xml = self.submit(
            {'equipment_type': equipment_type_map}, 'POST', url)

        return self.response(code, xml)