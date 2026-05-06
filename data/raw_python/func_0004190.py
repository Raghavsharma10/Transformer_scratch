def get_access(self, id_access):
        """Get Equipment Access by id.

        :return: Dictionary with following:

        ::

            {'equipamento_acesso':
            {'id_equipamento': < id_equipamento >,
            'fqdn': < fqdn >,
            'user': < user >,
            'pass': < pass >,
            'id_tipo_acesso': < id_tipo_acesso >,
            'enable_pass': < enable_pass >}}
        """

        if not is_valid_int_param(id_access):
            raise InvalidParameterError(u'Equipment Access ID is invalid.')

        url = 'equipamentoacesso/id/' + str(id_access) + '/'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)