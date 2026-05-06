def alterar(self, id_script, id_script_type, script, description, model=None):
        """Change Script from by the identifier.

        :param id_script: Identifier of the Script. Integer value and greater than zero.
        :param id_script_type: Identifier of the Script Type. Integer value and greater than zero.
        :param script: Script name. String with a minimum 3 and maximum of 40 characters
        :param description: Script description. String with a minimum 3 and maximum of 100 characters

        :return: None

        :raise InvalidParameterError: The identifier of Script, script Type, script or description is null and invalid.
        :raise RoteiroNaoExisteError: Script not registered.
        :raise TipoRoteiroNaoExisteError: Script Type not registered.
        :raise NomeRoteiroDuplicadoError: Script already registered with informed.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        if not is_valid_int_param(id_script):
            raise InvalidParameterError(u'The identifier of Script is invalid or was not informed.')

        script_map = dict()
        script_map['id_script_type'] = id_script_type
        script_map['script'] = script
        script_map['model'] = model
        script_map['description'] = description

        url = 'script/edit/' + str(id_script) + '/'

        code, xml = self.submit({'script': script_map}, 'PUT', url)

        return self.response(code, xml)