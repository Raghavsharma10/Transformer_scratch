def generate_key(self):
        """
        Ask mist.io to randomly generate a private ssh-key to be
        used with the creation of a new Key

        :returns: A string of a randomly generated ssh private key
        """
        req = self.request(self.uri + "/keys")
        private_key = req.post().json()
        return private_key['priv']