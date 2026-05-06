def create(self, cid, configData):
        """
        Create a new named (cid) configuration from a parameter dictionary (config_data).
        """
        configArgs = {'configId': cid, 'params': configData, 'force': True}
        cid = self.server.call('post', "/config/create", configArgs, forceText=True, headers=TextAcceptHeader)
        new_config = Config(cid, self.server)
        return new_config