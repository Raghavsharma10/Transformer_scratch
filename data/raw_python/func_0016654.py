def close(self):
        """Perform a clean shutdown.

        Waits for all the currently scheduled work to finish, kills the workers,
        and closes the connection pool.

        .. note:: Ensure your code does not submit new work while the connection is closing.

        Where possible, it is recommended you use a context manager (a :code:`with Client.from_config(...) as`
        construct) to ensure your code properly closes all resources.

        Examples:
            This example creates a client (based on an auto-detected configuration file), executes
            some code (represented by a placeholder comment), and then closes the client.

            >>> from dwave.cloud import Client
            >>> client = Client.from_config()
            >>> # code that uses client
            >>> client.close()

        """
        # Finish all the work that requires the connection
        _LOGGER.debug("Joining submission queue")
        self._submission_queue.join()
        _LOGGER.debug("Joining cancel queue")
        self._cancel_queue.join()
        _LOGGER.debug("Joining poll queue")
        self._poll_queue.join()
        _LOGGER.debug("Joining load queue")
        self._load_queue.join()

        # Send kill-task to all worker threads
        # Note: threads can't be 'killed' in Python, they have to die by
        # natural causes
        for _ in self._submission_workers:
            self._submission_queue.put(None)
        for _ in self._cancel_workers:
            self._cancel_queue.put(None)
        for _ in self._poll_workers:
            self._poll_queue.put((-1, None))
        for _ in self._load_workers:
            self._load_queue.put(None)

        # Wait for threads to die
        for worker in chain(self._submission_workers, self._cancel_workers,
                            self._poll_workers, self._load_workers):
            worker.join()

        # Close the requests session
        self.session.close()