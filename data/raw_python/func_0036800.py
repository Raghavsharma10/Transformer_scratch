def add_key(self, key_name, private):
        """
        Add a new key to mist.io

        :param key_name: Name of the new key (it will be used as the key's id as well).
        :param private: Private ssh-key in string format (see also generate_key() ).

        :returns: An updated list of added keys.
        """
        payload = {
            'name': key_name,
            'priv': private
        }

        data = json.dumps(payload)

        req = self.request(self.uri + '/keys', data=data)
        response = req.put().json()

        self.update_keys()
        return response