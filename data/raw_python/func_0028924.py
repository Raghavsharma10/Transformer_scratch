def set_state(self, state):
        """Set the runtime state of the Controller. Use the internal constants
        to ensure proper state values:

        - :attr:`Controller.STATE_INITIALIZING`
        - :attr:`Controller.STATE_ACTIVE`
        - :attr:`Controller.STATE_IDLE`
        - :attr:`Controller.STATE_SLEEPING`
        - :attr:`Controller.STATE_STOP_REQUESTED`
        - :attr:`Controller.STATE_STOPPING`
        - :attr:`Controller.STATE_STOPPED`

        :param int state: The runtime state
        :raises: ValueError

        """
        if state == self._state:
            return
        elif state not in self._STATES.keys():
            raise ValueError('Invalid state {}'.format(state))

        # Check for invalid transitions

        if self.is_waiting_to_stop and state not in [self.STATE_STOPPING,
                                                     self.STATE_STOPPED]:
            LOGGER.warning('Attempt to set invalid state while waiting to '
                           'shutdown: %s ', self._STATES[state])
            return

        elif self.is_stopping and state != self.STATE_STOPPED:
            LOGGER.warning('Attempt to set invalid post shutdown state: %s',
                           self._STATES[state])
            return

        elif self.is_running and state not in [self.STATE_ACTIVE,
                                               self.STATE_IDLE,
                                               self.STATE_SLEEPING,
                                               self.STATE_STOP_REQUESTED,
                                               self.STATE_STOPPING]:
            LOGGER.warning('Attempt to set invalid post running state: %s',
                           self._STATES[state])
            return

        elif self.is_sleeping and state not in [self.STATE_ACTIVE,
                                                self.STATE_IDLE,
                                                self.STATE_STOP_REQUESTED,
                                                self.STATE_STOPPING]:
            LOGGER.warning('Attempt to set invalid post sleeping state: %s',
                           self._STATES[state])
            return

        LOGGER.debug('State changed from %s to %s',
                     self._STATES[self._state], self._STATES[state])
        self._state = state