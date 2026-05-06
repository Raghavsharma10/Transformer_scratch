def dequeue(self):
        """
        Returns a :class:`~retask.task.Task` object from the queue. Returns ``None`` if the
        queue is empty.

        :return: :class:`~retask.task.Task` object from the queue

        If the queue is not connected then it will raise
        :class:`retask.ConnectionError`

        .. doctest::

           >>> from retask import Queue
           >>> q = Queue('test')
           >>> q.connect()
           True
           >>> t = q.dequeue()
           >>> print t.data
           {u'name': u'kushal'}

        """
        if not self.connected:
            raise ConnectionError('Queue is not connected')

        if self.rdb.llen(self._name) == 0:
            return None

        data = self.rdb.rpop(self._name)
        if not data:
            return None
        if isinstance(data, six.binary_type):
            data = six.text_type(data, 'utf-8', errors = 'replace')
        task = Task()
        task.__dict__ = json.loads(data)
        return task