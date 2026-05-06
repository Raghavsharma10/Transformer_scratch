def set_state_transition_func(self, func):
        """THIS FUNCTION MUST BE CALLED BEFORE CALLING
        freshroastsr700.auto_connect().

        Set, or re-set, the state transition function callback.
        The supplied function will be called from a separate thread within
        freshroastsr700, triggered by a separate, internal child process.
        This function will fail if the freshroastsr700 device is already
        connected to hardware, because by that time, the timer process
        and thread have already been spawned.

        Args:
            state_transition_func (func): the function to call for every
            state transition.  A state transition occurs whenever the
            freshroastsr700's time_remaining value counts down to 0.

        Returns:
            nothing
       """
        if self._connected.value:
            logging.error("freshroastsr700.set_state_transition_func must be "
                          "called before freshroastsr700.auto_connect()."
                          " Not registering func.")
            return False
        # no connection yet. so OK to set func pointer
        self._create_state_transition_system(func)
        return True