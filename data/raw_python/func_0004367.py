def get_all(self):
        """Return all equipments in database

        :return: Dictionary with the following structure:

        ::
            {'equipaments': {'name' :< name_equipament >}, {... demais equipamentos ...} }

        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        url = 'equipment/all'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)