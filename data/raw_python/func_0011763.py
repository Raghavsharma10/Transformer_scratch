def state_transition_run(self, event_to_wait_on):
        """This is the thread that listens to an event from
           the timer process to execute the state_transition_func callback
           in the context of the main process.
           """
        # with the daemon=Turue setting, this thread should
        # quit 'automatically'
        while event_to_wait_on.wait():
            event_to_wait_on.clear()
            if self.state_transition_callback_kill_event.is_set():
                return
            self.state_transition_func()