def wait(self, wait_time=0):
        """
        Returns a :class:`~retask.task.Task` object from the queue. Returns ``False`` if it timeouts.

        :arg wait_time: Time in seconds to wait, default is infinite.

        :return: :class:`~retask.task.Task` object from the queue or False if it timeouts.

        .. doctest::

           >>> from retask import Queue
           >>> q = Queue('test')
           >>> q.connect()
           True
           >>> task = q.wait()
           >>> print task.data
           {u'name': u'kushal'}

        .. note::

            This is a blocking call, you can specity wait_time argument for timeout.

        """
        if not self.connected:
            raise ConnectionError('Queue is not connected')

        data = self.rdb.brpop(self._name, wait_time)
        if data:
            task = Task()
            task.__dict__ = json.loads(data[1])
            return task
        else:
            return False