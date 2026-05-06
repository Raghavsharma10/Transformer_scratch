def _handle_terminate(self, signal_number, _):
        """Handle a signal to terminate."""
        signal_names = {
            signal.SIGINT: 'SIGINT',
            signal.SIGQUIT: 'SIGQUIT',
            signal.SIGTERM: 'SIGTERM',
        }
        message = 'Terminated by {name} ({number})'.format(
            name=signal_names[signal_number], number=signal_number)
        self._shutdown(message, code=128+signal_number)