def locations(self):
        """
        Available locations to be used when creating a new machine.

        :returns: A list of available locations.
        """
        req = self.request(self.mist_client.uri+'/clouds/'+self.id+'/locations')
        locations = req.get().json()
        return locations