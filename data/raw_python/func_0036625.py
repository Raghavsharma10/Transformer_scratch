def sizes(self):
        """
        Available machine sizes to be used when creating a new machine.

        :returns: A list of available machine sizes.
        """
        req = self.request(self.mist_client.uri+'/clouds/'+self.id+'/sizes')
        sizes = req.get().json()
        return sizes