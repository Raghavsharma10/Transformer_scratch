def inserir(self, id_equipment, id_script):
        """Inserts a new Related Equipment with Script and returns its identifier

        :param id_equipment: Identifier of the Equipment. Integer value and greater than zero.
        :param id_script: Identifier of the Script. Integer value and greater than zero.

        :return: Dictionary with the following structure:

        ::

          {'equipamento_roteiro': {'id': < id_equipment_script >}}

        :raise InvalidParameterError: The identifier of Equipment or Script is null and invalid.
        :raise RoteiroNaoExisteError: Script not registered.
        :raise EquipamentoNaoExisteError: Equipment not registered.
        :raise EquipamentoRoteiroError: Equipment is already associated with the script.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response
        """
        equipment_script_map = dict()
        equipment_script_map['id_equipment'] = id_equipment
        equipment_script_map['id_script'] = id_script

        code, xml = self.submit(
            {'equipment_script': equipment_script_map}, 'POST', 'equipmentscript/')

        return self.response(code, xml)