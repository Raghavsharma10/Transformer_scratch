def public(self):
        """
        Return the public ssh-key

        :returns: The public ssh-key as string
        """
        req = self.request(self.mist_client.uri+'/keys/'+self.id+"/public")
        public = req.get().json()
        return public