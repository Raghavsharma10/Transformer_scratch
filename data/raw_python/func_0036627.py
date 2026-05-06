def images(self):
        """
        Available images to be used when creating a new machine.

        :returns: A list of all available images.
        """
        req = self.request(self.mist_client.uri+'/clouds/'+self.id+'/images')
        images = req.get().json()
        return images