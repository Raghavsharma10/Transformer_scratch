def urlinfo(self, domain, response_group = URLINFO_RESPONSE_GROUPS):
        '''
        Provide information about supplied domain as specified by the response group
        :param domain: Any valid URL
        :param response_group: Any valid urlinfo response group
        :return: Traffic and/or content data of the domain in XML format
        '''
        params = {
            'Action': "UrlInfo",
            'Url': domain,
            'ResponseGroup': response_group
        }

        url, headers = self.create_v4_signature(params)
        return self.return_output(url, headers)