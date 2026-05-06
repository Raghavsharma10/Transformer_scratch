def list_acl_path(self):
        """Get all distinct acl paths.

        :return: Dictionary with the following structure:

        ::

            {'acl_paths': [
             < acl_path >,
             ... ]}



        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        url = 'environment/acl_path/'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)