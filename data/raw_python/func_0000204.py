def list(self, obj, filter=False, only_id=False, limit=20):
        """ Function list
        Get the list of an object

        @param obj: object name ('hosts', 'puppetclasses'...)
        @param filter: filter for objects
        @param only_id: boolean to only return dict with name/id
        @return RETURN: the list of the object
        """
        self.url = '{}{}/?per_page={}'.format(self.base_url, obj, limit)
        self.method = 'GET'
        if filter:
            self.url += '&search={}'.format(filter)
        self.resp = requests.get(url=self.url, auth=self.auth,
                                 headers=self.headers, cert=self.ca_cert)
        if only_id:
            if self.__process_resp__(obj) is False:
                return False
            if type(self.res['results']) is list:
                return dict((x['name'], x['id']) for x in self.res['results'])
            elif type(self.res['results']) is dict:
                r = {}
                for v in self.res['results'].values():
                    for vv in v:
                        r[vv['name']] = vv['id']
                return r
            else:
                return False
        else:
            return self.__process_resp__(obj)