def restart(self, hard=False, timeout=30, force=True):
        """
        Restart the server. By default, issue a soft restart with a timeout of 30s
        and a hard restart after the timeout.

        After the a timeout a hard restart is performed if the server has not stopped.

        Note: API responds immediately (unlike in start), with state: started.
        This client will, however, set state as 'maintenance' to signal that the server is neither
        started nor stopped.
        """
        body = dict()
        body['restart_server'] = {
            'stop_type': 'hard' if hard else 'soft',
            'timeout': '{0}'.format(timeout),
            'timeout_action': 'destroy' if force else 'ignore'
        }

        path = '/server/{0}/restart'.format(self.uuid)
        self.cloud_manager.post_request(path, body)
        object.__setattr__(self, 'state', 'maintenance')