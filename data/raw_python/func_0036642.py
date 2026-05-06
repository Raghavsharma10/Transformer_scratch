def set_default(self):
        """
        Set this key as the default key

        :returns: An updated list of added keys
        """
        req = self.request(self.mist_client.uri+'/keys/'+self.id)
        req.post()
        self.is_default = True
        self.mist_client.update_keys()