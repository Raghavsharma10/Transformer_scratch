def stop_and_destroy(self, sync=True):
        """
        Destroy a server and its storages. Stops the server before destroying.

        Syncs the server state from the API, use sync=False to disable.
        """
        def _self_destruct():
            """destroy the server and all storages attached to it."""

            # try_it_n_times util is used as a convenience because
            # Servers and Storages can fluctuate between "maintenance" and their
            # original state due to several different reasons especially when
            # destroying infrastructure.

            # first destroy server
            try_it_n_times(operation=self.destroy,
                           expected_error_codes=['SERVER_STATE_ILLEGAL'],
                           custom_error='destroying server failed')

            # storages may be deleted instantly after server DELETE
            for storage in self.storage_devices:
                try_it_n_times(operation=storage.destroy,
                               expected_error_codes=['STORAGE_STATE_ILLEGAL'],
                               custom_error='destroying storage failed')

        if sync:
            self.populate()

        # server is either starting or stopping (or error)
        if self.state in ['maintenance', 'error']:
            self._wait_for_state_change(['stopped', 'started'])

        if self.state == 'started':
            try_it_n_times(operation=self.stop,
                           expected_error_codes=['SERVER_STATE_ILLEGAL'],
                           custom_error='stopping server failed')

            self._wait_for_state_change(['stopped'])

        if self.state == 'stopped':
            _self_destruct()
        else:
            raise Exception('unknown server state: ' + self.state)