def shutdown(self, hard=False, timeout=30):
        """
        Shutdown/stop the server. By default, issue a soft shutdown with a timeout of 30s.

        After the a timeout a hard shutdown is performed if the server has not stopped.

        Note: API responds immediately (unlike in start), with state: started.
        This client will, however, set state as 'maintenance' to signal that the server is neither
        started nor stopped.
        """
        body = dict()
        body['stop_server'] = {
            'stop_type': 'hard' if hard else 'soft',
            'timeout': '{0}'.format(timeout)
        }

        path = '/server/{0}/stop'.format(self.uuid)
        self.cloud_manager.post_request(path, body)
        object.__setattr__(self, 'state', 'maintenance')