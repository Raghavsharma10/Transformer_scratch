def send(self, task, result, expire=60):
        """
        Sends the result back to the producer. This should be called if only you
        want to return the result in async manner.

        :arg task: ::class:`~retask.task.Task` object
        :arg result: Result data to be send back. Should be in JSON serializable.
        :arg expire: Time in seconds after the key expires. Default is 60 seconds.
        """
        self.rdb.lpush(task.urn, json.dumps(result))
        self.rdb.expire(task.urn, expire)