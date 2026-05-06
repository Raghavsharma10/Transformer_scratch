def inserir(self, id_script_type, script, model, description):
        """Inserts a new Script and returns its identifier.

        :param id_script_type: Identifier of the Script Type. Integer value and greater than zero.
        :param script: Script name. String with a minimum 3 and maximum of 40 characters
        :param description: Script description. String with a minimum 3 and maximum of 100 characters

        :return: Dictionary with the following structure:

        ::

            {'script': {'id': < id_script >}}

        :raise InvalidParameterError: The identifier of Script Type, script or description is null and invalid.
        :raise TipoRoteiroNaoExisteError: Script Type not registered.
        :raise NomeRoteiroDuplicadoError: Script already registered with informed.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        script_map = dict()
        script_map['id_script_type'] = id_script_type
        script_map['script'] = script
        script_map['model'] = model
        script_map['description'] = description

        code, xml = self.submit({'script': script_map}, 'POST', 'script/')

        return self.response(code, xml)