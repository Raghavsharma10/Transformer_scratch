def kill_thread(self, name):
        """
        Joins the thread in the `thread_pool` dict with the given `name` key.
        """
        if name not in self.thread_pool:
            return

        self.thread_pool[name].join()
        del self.thread_pool[name]