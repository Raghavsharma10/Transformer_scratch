def kill_process(self, payload):
        """Pause the daemon and kill all processes or kill a specific process."""
        # Kill specific processes, if `keys` is given in the payload
        kill_signal = signals[payload['signal'].lower()]
        kill_shell = payload.get('all', False)
        if payload.get('keys'):
            succeeded = []
            failed = []
            for key in payload.get('keys'):
                success = self.process_handler.kill_process(key, kill_signal, kill_shell)
                if success:
                    succeeded.append(str(key))
                else:
                    failed.append(str(key))

            message = ''
            if len(succeeded) > 0:
                message += "Signal '{}' sent to processes: {}.".format(payload['signal'], ', '.join(succeeded))
                status = 'success'
            if len(failed) > 0:
                message += '\nNo running process for keys: {}'.format(', '.join(failed))
                status = 'error'

            answer = {'message': message.strip(), 'status': status}

        # Kill all processes and the daemon
        else:
            self.process_handler.kill_all(kill_signal, kill_shell)
            if kill_signal == signal.SIGINT or \
               kill_signal == signal.SIGTERM or \
               kill_signal == signal.SIGKILL:
                self.paused = True
            answer = {'message': 'Signal send to all processes.',
                      'status': 'success'}
        return answer