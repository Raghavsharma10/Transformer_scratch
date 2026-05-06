def enqueue(self, task):
        """
        Enqueues the given :class:`~retask.task.Task` object to the queue and returns
        a :class:`~retask.queue.Job` object.

        :arg task: ::class:`~retask.task.Task` object

        :return: :class:`~retask.queue.Job` object

        If the queue is not connected then it will raise
        :class:`retask.ConnectionError`.

        .. doctest::

           >>> from retask import Queue
           >>> q = Queue('test')
           >>> q.connect()
           True
           >>> from retask.task import Task
           >>> task = Task({'name':'kushal'})
           >>> job = q.enqueue(task)

        """
        if not self.connected:
            raise ConnectionError('Queue is not connected')

        try:
            #We can set the value to the queue
            job = Job(self.rdb)
            task.urn = job.urn
            text = json.dumps(task.__dict__)
            self.rdb.lpush(self._name, text)
        except Exception as err:
            return False
        return job