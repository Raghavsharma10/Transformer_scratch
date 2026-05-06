def get(self, obj, id, sub_object=None):
        """ Function get
        Get an object by id

        @param obj: object name ('hosts', 'puppetclasses'...)
        @param id: the id of the object (name or id)
        @return RETURN: the targeted object
        """
        self.url = '{}{}/{}'.format(self.base_url, obj, id)
        self.method = 'GET'
        if sub_object:
            self.url += '/' + sub_object
        self.resp = requests.get(url=self.url, auth=self.auth,
                                 headers=self.headers, cert=self.ca_cert)
        if self.__process_resp__(obj):
            return self.res
        return False