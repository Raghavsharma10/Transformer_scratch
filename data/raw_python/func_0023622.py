def stop_daemon(self, payload=None):
        """Kill current processes and initiate daemon shutdown.

        The daemon will shut down after a last check on all killed processes.
        """
        kill_signal = signals['9']
        self.process_handler.kill_all(kill_signal, True)
        self.running = False

        return {'message': 'Pueue daemon shutting down',
                'status': 'success'}