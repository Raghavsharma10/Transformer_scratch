def wait(self, wait_time=0):
        """
        Blocking call to check if the worker returns the result. One can use
        job.result after this call returns ``True``.

        :arg wait_time: Time in seconds to wait, default is infinite.

        :return: `True` or `False`.

        .. note::

            This is a blocking call, you can specity wait_time argument for timeout.

        """
        if self.__result:
            return True
        data = self.rdb.brpop(self.urn, wait_time)
        if data:
            self.rdb.delete(self.urn)
            data = json.loads(data[1])
            self.__result = data
            return True
        else:
            return False