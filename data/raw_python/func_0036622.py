def delete(self):
        """
        Delete the cloud from the list of added clouds in mist.io service.

        :returns: A list of mist.clients' updated clouds.
        """
        req = self.request(self.mist_client.uri + '/clouds/' + self.id)
        req.delete()
        self.mist_client.update_clouds()