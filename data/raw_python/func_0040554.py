def start(self):
        """this function will start the queing thread that executes the
        iterator and feeds jobs into the queue.  It also starts the worker
        threads that just sit and wait for items to appear on the queue. This
        is a non blocking call, so the executing thread is free to do other
        things while the other threads work."""
        self.logger.debug('start')
        # start each of the task threads.
        for x in range(self.number_of_threads):
            # each thread is given the config object as well as a reference to
            # this manager class.  The manager class is where the queue lives
            # and the task threads will refer to it to get their next jobs.
            new_thread = TaskThread(self.config, self.task_queue)
            self.thread_list.append(new_thread)
            new_thread.start()
        self.queuing_thread = threading.Thread(
          name="QueuingThread",
          target=self._queuing_thread_func
        )
        self.queuing_thread.start()