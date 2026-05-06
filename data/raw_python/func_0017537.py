def cat_browse(self, domain, path, response_group=CATEGORYBROWSE_RESPONSE_GROUPS, descriptions='True'):
        '''
        Provide category browse information of specified domain
        :param domain: Any valid URL
        :param path: Valid category path
        :param response_group: Any valid traffic history response group
        :return: Traffic and/or content data of the domain in XML format
        '''
        params = {
            'Action': "CategoryListings",
            'ResponseGroup': 'Listings',
            'Path': quote(path),
            'Descriptions': descriptions
        }

        url, headers = self.create_v4_signature(params)
        return self.return_output(url, headers)