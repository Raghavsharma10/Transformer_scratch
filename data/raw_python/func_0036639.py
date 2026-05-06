def private(self):
        """
        Return the private ssh-key

        :returns: The private ssh-key as string
        """
        req = self.request(self.mist_client.uri+'/keys/'+self.id+"/private")
        private = req.get().json()
        return private