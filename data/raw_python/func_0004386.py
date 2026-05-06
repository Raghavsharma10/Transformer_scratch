def insert_with_ip_range(
            self,
            id_l3_group,
            id_logical_environment,
            id_division,
            id_ip_config,
            link,
            id_filter=None):
        """Insert new environment with ip config and returns your id.

        :param id_l3_group: Layer 3 Group ID.
        :param id_logical_environment: Logical Environment ID.
        :param id_division: Data Center Division ID.
        :param id_filter: Filter identifier.
        :param id_ip_config: IP Configuration ID.
        :param link: Link.

        :return: Following dictionary: {'ambiente': {'id': < id >}}

        :raise ConfigEnvironmentDuplicateError: Error saving duplicate Environment Configuration.
        :raise InvalidParameterError: Some parameter was invalid.
        :raise GrupoL3NaoExisteError: Layer 3 Group not found.
        :raise AmbienteLogicoNaoExisteError: Logical Environment not found.
        :raise DivisaoDcNaoExisteError: Data Center Division not found.
        :raise AmbienteDuplicadoError: Environment with this parameters already exists.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        environment_map = dict()
        environment_map['id_grupo_l3'] = id_l3_group
        environment_map['id_ambiente_logico'] = id_logical_environment
        environment_map['id_divisao'] = id_division
        environment_map['id_filter'] = id_filter
        environment_map['id_ip_config'] = id_ip_config
        environment_map['link'] = link

        code, xml = self.submit(
            {'ambiente': environment_map}, 'POST', 'ambiente/ipconfig/')

        return self.response(code, xml)