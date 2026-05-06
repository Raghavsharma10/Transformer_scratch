def list_with_usergroup(self):
        """List all users and their user groups.
        is_more -If  more than 3 of groups of users or no, to control expansion Screen.

        :return: Dictionary with the following structure:

        ::

            {'usuario': [{'nome': < nome >,
            'id': < id >,
            'pwd': < pwd >,
            'user': < user >,
            'ativo': < ativo >,
            'email': < email >,
            'is_more': <True ou False>,
            'grupos': [nome_grupo, ...more user groups...]}, ...more user...]}


        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        url = 'usuario/get/'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)