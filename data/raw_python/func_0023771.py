def _wait_for_state_change(self, target_states, update_interval=10):
        """
        Blocking wait until target_state reached. update_interval is in seconds.

        Warning: state change must begin before calling this method.
        """
        while self.state not in target_states:
            if self.state == 'error':
                raise Exception('server is in error state')

            # update server state every 10s
            sleep(update_interval)
            self.populate()