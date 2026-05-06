def wait_for_up(self, timeout=40):
        """ Wait until port is up and running, including all parameters (admin state, oper state, license etc.).

        :param timeout: max time to wait for port up.
        """

        self.wait_for_states(timeout, 'up')
        connectionStatus = self.get_attribute('connectionStatus').strip()
        if connectionStatus.split(':')[0] != self.get_attribute('assignedTo').split(':')[0]:
            raise TgnError('Failed to reach up state, port connection status is {} after {} seconds'.
                           format(connectionStatus, timeout))