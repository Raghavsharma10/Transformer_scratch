def delete(self):
        """
        Delete this key from mist.io

        :returns: An updated list of added keys
        """
        req = self.request(self.mist_client.uri+'/keys/'+self.id)
        req.delete()
        self.mist_client.update_keys()