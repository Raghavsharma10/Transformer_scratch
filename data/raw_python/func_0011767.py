def _start_connect(self, connect_type):
        """Starts the connection process, as called (internally)
        from the user context, either from auto_connect() or connect().
        Never call this from the _comm() process context.
        """
        if self._connect_state.value != self.CS_NOT_CONNECTED:
            # already done or in process, assume success
            return

        self._connected.value = 0
        self._connect_state.value = self.CS_ATTEMPTING_CONNECT
        # tell comm process to attempt connection
        self._attempting_connect.value = connect_type

        # EXTREMELY IMPORTANT - for this to work at all in Windows,
        # where the above processes are spawned (vs forked in Unix),
        # the thread objects (as sattributes of this object) must be
        # assigned to this object AFTER we have spawned the processes.
        # That way, multiprocessing can pickle the freshroastsr700
        # successfully. (It can't pickle thread-related stuff.)
        if self.update_data_func is not None:
            # Need to launch the thread that will listen to the event
            self._create_update_data_system(
                None, setFunc=False, createThread=True)
            self.update_data_thread.start()
        if self.state_transition_func is not None:
            # Need to launch the thread that will listen to the event
            self._create_state_transition_system(
                None, setFunc=False, createThread=True)
            self.state_transition_thread.start()