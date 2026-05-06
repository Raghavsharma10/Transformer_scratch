def wait_for_states(self, timeout=40, *states):
        """ Wait until port reaches one of the requested states.

        :param timeout: max time to wait for requested port states.
        """

        state = self.get_attribute('state')
        for _ in range(timeout):
            if state in states:
                return
            time.sleep(1)
            state = self.get_attribute('state')
        raise TgnError('Failed to reach states {}, port state is {} after {} seconds'.format(states, state, timeout))