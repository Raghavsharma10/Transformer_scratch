def configuration_list_all(self, environment_id):
        """
        List all prefix configurations by environment in DB

        :return: Following dictionary:

        ::

            {'lists_configuration': [{
            'id': <id_ipconfig>,
            'subnet': <subnet>,
            'type': <type>,
            'new_prefix': <new_prefix>,
            }, ... ]}


        :raise InvalidValueError: Invalid ID for Environment.
        :raise AmbienteNotFoundError: Environment not registered.
        :raise DataBaseError: Failed into networkapi access data base.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        data = dict()
        data["environment_id"] = environment_id

        url = ("environment/configuration/list/%(environment_id)s/" % data)

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml, force_list=['lists_configuration'])