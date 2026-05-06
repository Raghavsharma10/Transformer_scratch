def traffichistory(self, domain, response_group=TRAFFICINFO_RESPONSE_GROUPS, myrange=31, start=20070801):
        '''
        Provide traffic history of supplied domain
        :param domain: Any valid URL
        :param response_group: Any valid traffic history response group
        :return: Traffic and/or content data of the domain in XML format
        '''
        params = {
            'Action': "TrafficHistory",
            'Url': domain,
            'ResponseGroup': response_group,
            'Range': myrange,
            'Start': start,
        }

        url, headers = self.create_v4_signature(params)
        return self.return_output(url, headers)