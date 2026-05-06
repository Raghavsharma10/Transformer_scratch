def find(self, obj):
        """Returns the index of the given object in the queue, it might be string
        which will be searched inside each task.

        :arg obj: object we are looking

        :return: -1 if the object is not found or else the location of the task
        """
        if not self.connected:
            raise ConnectionError('Queue is not connected')

        data = self.rdb.lrange(self._name, 0, -1)
        for i, datum in enumerate(data):
            if datum.find(str(obj)) != -1:
                return i
        return -1