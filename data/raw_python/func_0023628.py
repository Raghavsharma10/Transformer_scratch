def start(self, payload):
        """Start the daemon and all processes or only specific processes."""
        # Start specific processes, if `keys` is given in the payload
        if payload.get('keys'):
            succeeded = []
            failed = []
            for key in payload.get('keys'):
                success = self.process_handler.start_process(key)
                if success:
                    succeeded.append(str(key))
                else:
                    failed.append(str(key))

            message = ''
            if len(succeeded) > 0:
                message += 'Started processes: {}.'.format(', '.join(succeeded))
                status = 'success'
            if len(failed) > 0:
                message += '\nNo paused, queued or stashed process for keys: {}'.format(', '.join(failed))
                status = 'error'

            answer = {'message': message.strip(), 'status': status}

        # Start a all processes and the daemon
        else:
            self.process_handler.start_all()
            if self.paused:
                self.paused = False
                answer = {'message': 'Daemon and all processes started.',
                          'status': 'success'}
            else:
                answer = {'message': 'Daemon already running, starting all processes.',
                          'status': 'success'}
        return answer