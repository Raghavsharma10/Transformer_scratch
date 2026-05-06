def delete(self, obj, id):
        """ Function delete
        Delete an object by id

        @param obj: object name ('hosts', 'puppetclasses'...)
        @param id: the id of the object (name or id)
        @return RETURN: the server response
        """
        self.url = '{}{}/{}'.format(self.base_url, obj, id)
        self.method = 'DELETE'
        self.resp = requests.delete(url=self.url,
                                    auth=self.auth,
                                    headers=self.headers, cert=self.ca_cert)
        return self.__process_resp__(obj)