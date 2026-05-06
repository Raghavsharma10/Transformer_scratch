def json(self):
        """
        return __cached_json, if accessed withing 300 ms.
        This allows to optimize calls when many parameters of entity requires withing short time.
        """

        if self.fresh():
            return self.__cached_json
        # noinspection PyAttributeOutsideInit
        self.__last_read_time = time.time()
        self.__cached_json = self._router.get_instance(org_id=self.organizationId, instance_id=self.instanceId).json()

        return self.__cached_json