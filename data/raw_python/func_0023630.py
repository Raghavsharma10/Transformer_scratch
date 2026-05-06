def edit_command(self, payload):
        """Edit the command of a specific entry."""
        key = payload['key']
        command = payload['command']
        if self.queue[key]:
            if self.queue[key]['status'] in ['queued', 'stashed']:
                self.queue[key]['command'] = command
                answer = {'message': 'Command updated', 'status': 'error'}
            else:
                answer = {'message': "Entry is not 'queued' or 'stashed'",
                          'status': 'error'}
        else:
            answer = {'message': 'No entry with this key', 'status': 'error'}

        # Pause all processes and the daemon
        return answer