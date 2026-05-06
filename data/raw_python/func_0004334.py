def get_by_user_ldap(self, user_name):
        """Get user by the ldap name.
        is_more -If  more than 3 of groups of users or no, to control expansion Screen.

        :return: Dictionary with the following structure:

        ::

            {'usuario': [{'nome': < nome >,
            'id': < id >,
            'pwd': < pwd >,
            'user': < user >,
            'ativo': < ativo >,
            'email': < email >,
            'grupos': [nome_grupo, ...more user groups...],
            'user_ldap': < user_ldap >}}


        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        url = 'user/get/ldap/' + str(user_name) + '/'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)