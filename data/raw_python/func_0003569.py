def parallelize(self, seconds_to_wait=2):
        """Start a parallel thread for receiving messages.

        If :meth:`start` was no called before, start will be called in the
        thread.
        The thread calls :meth:`receive_message` until the :attr:`state`
        :meth:`~AYABInterface.communication.states.State.is_connection_closed`.

        :param float seconds_to_wait: A time in seconds to wait with the
          parallel execution. This is useful to allow the controller time to
          initialize.

        .. seealso:: :attr:`lock`, :meth:`runs_in_parallel`
        """
        with self.lock:
            thread = Thread(target=self._parallel_receive_loop,
                            args=(seconds_to_wait,))
            thread.deamon = True
            thread.start()
            self._thread = thread