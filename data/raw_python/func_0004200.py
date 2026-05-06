def create(self, pools):
        """
            Create Pools Running Script And Update to Created

            :param pools: List of pools

            :return: None on success

            :raise PoolDoesNotExistException
            :raise ScriptCreatePoolException
            :raise InvalidIdPoolException
            :raise NetworkAPIException
        """

        data = dict()
        data["pools"] = pools

        uri = "api/pools/v2/"

        return self.put(uri, data)