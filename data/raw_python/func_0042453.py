def close(self):
        """
        OPTIONAL COMMIT-AND-CLOSE
        IF THIS IS NOT DONE, THEN THE THREAD THAT SPAWNED THIS INSTANCE
        :return:
        """
        self.closed = True
        signal = _allocate_lock()
        signal.acquire()
        self.queue.add(CommandItem(COMMIT, None, signal, None, None))
        signal.acquire()
        self.worker.please_stop.go()
        return