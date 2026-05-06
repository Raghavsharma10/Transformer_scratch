def run(self):
        """The core method for starting the application. Will setup logging,
        toggle the runtime state flag, block on loop, then call shutdown.

        Redefine this method if you intend to use an IO Loop or some other
        long running process.

        """
        LOGGER.info('%s v%s started', self.APPNAME, self.VERSION)
        self.setup()
        while not any([self.is_stopping, self.is_stopped]):
            self.set_state(self.STATE_SLEEPING)
            try:
                signum = self.pending_signals.get(True, self.wake_interval)
            except queue.Empty:
                pass
            else:
                self.process_signal(signum)
                if any([self.is_stopping, self.is_stopped]):
                    break
            self.set_state(self.STATE_ACTIVE)
            self.process()