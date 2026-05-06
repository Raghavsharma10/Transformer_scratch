def retrieve(self):
        """
        Retrieve a result from executing a task. Note that tasks are executed
        in order and that if the next task has not yet completed, this call
        will block until the result is available.

        Returns
        -------
        A result from the result buffer.
        """
        if len(self.__result_buffer) > 0:
            res = self.__result_buffer.popleft()
            value = res.get()
        else:
            return None

        self.__populate_buffer()

        return value