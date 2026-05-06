def disable(self, ids):
        """
            Disable Pool Members Running Script

            :param ids: List of ids

            :return: None on success

            :raise PoolMemberDoesNotExistException
            :raise InvalidIdPoolMemberException
            :raise ScriptDisablePoolException
            :raise NetworkAPIException
        """

        data = dict()
        data["ids"] = ids

        uri = "api/pools/disable/"

        return self.post(uri, data)