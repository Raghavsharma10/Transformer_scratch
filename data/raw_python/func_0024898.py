def _get_cloud_foundry_config(self):
        """
        Reads the local cf CLI cache stored in the users
        home directory.
        """
        config = os.path.expanduser(self.config_file)
        if not os.path.exists(config):
            raise CloudFoundryLoginError('You must run `cf login` to authenticate')

        with open(config, "r") as data:
            return json.load(data)