def rename(self, new_name):
        """
        Rename a key

        :param new_name: New name for the key (will also serve as the key's id)
        :returns: An updated list of added keys
        """
        payload = {
            'new_name': new_name
        }
        data = json.dumps(payload)
        req = self.request(self.mist_client.uri+'/keys/'+self.id, data=data)
        req.put()
        self.id = new_name
        self.mist_client.update_keys()