def gevent_spawn(self):
        """ Spawn worker threads (using gevent) """
        monkey.patch_all(thread=False)
        joinall([spawn(self.gevent_worker) for x in range(self.queue_worker_amount)])