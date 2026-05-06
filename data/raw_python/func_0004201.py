def enable(self, ids):
        """
            Enable Pool Members Running Script

            :param ids: List of ids

            :return: None on success

            :raise PoolMemberDoesNotExistException
            :raise InvalidIdPoolMemberException
            :raise ScriptEnablePoolException
            :raise NetworkAPIException
        """

        data = dict()
        data["ids"] = ids

        uri = "api/pools/enable/"

        return self.post(uri, data)