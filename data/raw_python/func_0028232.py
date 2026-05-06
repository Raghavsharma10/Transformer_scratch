def start_watcher_thread(self):
        """
        Start watcher thread.

        :return:
            Watcher thread object.
        """
        # Create watcher thread
        watcher_thread = threading.Thread(target=self.run_watcher)

        # If the reload mode is `spawn_wait`
        if self._reload_mode == self.RELOAD_MODE_V_SPAWN_WAIT:
            # Use non-daemon thread
            daemon = False

        # If the reload mode is not `spawn_wait`
        else:
            # Use daemon thread
            daemon = True

        # Set whether the thread is daemon
        watcher_thread.setDaemon(daemon)

        # Start watcher thread
        watcher_thread.start()

        # Return watcher thread
        return watcher_thread