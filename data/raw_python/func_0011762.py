def update_data_run(self, event_to_wait_on):
        """This is the thread that listens to an event from
           the comm process to execute the update_data_func callback
           in the context of the main process.
           """
        # with the daemon=Turue setting, this thread should
        # quit 'automatically'
        while event_to_wait_on.wait():
            event_to_wait_on.clear()
            if self.update_data_callback_kill_event.is_set():
                return
            self.update_data_func()