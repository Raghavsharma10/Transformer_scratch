def pause(self, payload):
        """Start the daemon and all processes or only specific processes."""
        # Pause specific processes, if `keys` is given in the payload
        if payload.get('keys'):
            succeeded = []
            failed = []
            for key in payload.get('keys'):
                success = self.process_handler.pause_process(key)
                if success:
                    succeeded.append(str(key))
                else:
                    failed.append(str(key))

            message = ''
            if len(succeeded) > 0:
                message += 'Paused processes: {}.'.format(', '.join(succeeded))
                status = 'success'
            if len(failed) > 0:
                message += '\nNo running process for keys: {}'.format(', '.join(failed))
                status = 'error'

            answer = {'message': message.strip(), 'status': status}

        # Pause all processes and the daemon
        else:
            if payload.get('wait'):
                self.paused = True
                answer = {'message': 'Pausing daemon, but waiting for processes to finish.',
                          'status': 'success'}
            else:
                self.process_handler.pause_all()
                if not self.paused:
                    self.paused = True
                    answer = {'message': 'Daemon and all processes paused.',
                              'status': 'success'}
                else:
                    answer = {'message': 'Daemon already paused, pausing all processes anyway.',
                              'status': 'success'}

        return answer