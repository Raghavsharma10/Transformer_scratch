def stop(self):
        """
        Method for shutting down the watcher.

        All config file observers are stopped and their threads joined, along
        with the worker thread pool.
        """
        self.shutdown.set()

        for monitor in self.observers:
            monitor.stop()

        self.wind_down()

        for monitor in self.observers:
            monitor.join()

        for thread in self.thread_pool.values():
            thread.join()

        self.work_pool.shutdown()