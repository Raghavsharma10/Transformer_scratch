def _reverse_queue(self):
        u"""When socket.timeout has occurred for Zabbix server,
        this method is called.
        Enqueue items in self.pool[].
        """

        while self.pool:
            item = self.pool.pop()
            self.queue.put(item, block=False)