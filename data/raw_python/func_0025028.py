def create_user(self, username, password, family_name, given_name, primary_email,
            details={}):
        """
        Creates a new user account with the required details.

        ::

            create_user('j12y', 'my-secret', 'Delancey', 'Jayson', 'volcano@ge.com')

        """
        self.assert_has_permission('scim.write')

        data = {
            'userName': username,
            'password': password,
            'name': {
                'familyName': family_name,
                'givenName': given_name,
                },
            'emails': [{
                'value': primary_email,
                'primary': True,
                }]
            }

        if details:
            data.update(details)

        return self._post_user(data)