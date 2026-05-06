def wait_until_complete(self, timeout=None):
        """Wait until sequencer is finished.

        This method blocks your application until the sequencer has completed
        its operation.  It returns once the sequencer has finished.

        Arguments:
        timeout -- Optional.  Seconds to wait for sequencer to finish.  If this
                   time is exceeded, then an exception is raised.

        Return:
        Sequencer testState value.

        """
        timeout_at = None
        if timeout:
            timeout_at = time.time() + int(timeout)

        sequencer = self.get('system1', 'children-sequencer')
        while True:
            cur_test_state = self.get(sequencer, 'state')
            if 'PAUSE' in cur_test_state or 'IDLE' in cur_test_state:
                break
            time.sleep(2)
            if timeout_at and time.time() >= timeout_at:
                raise RuntimeError('wait_until_complete timed out after %s sec'
                                   % timeout)

        return self.get(sequencer, 'testState')