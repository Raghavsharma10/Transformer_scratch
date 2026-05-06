def add_thread(self, func, interval):
        """
        Creates a thread, starts it and then adds it to the thread pool.

        Func: Same as in the Thread class.
        Interval: Same as in the Thread class.
        """
        t = Thread(func, interval, self.output_dict)
        t.start()
        self._thread_pool.append(t)