def remove(self, pools):
        """
            Remove Pools Running Script And Update to Not Created

            :param ids: List of ids

            :return: None on success

            :raise ScriptRemovePoolException
            :raise InvalidIdPoolException
            :raise NetworkAPIException
        """

        data = dict()
        data["pools"] = pools

        uri = "api/pools/v2/"

        return self.delete(uri, data)