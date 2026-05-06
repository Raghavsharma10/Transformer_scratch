def restart(self, payload):
        """Restart the specified entries."""
        succeeded = []
        failed = []
        for key in payload['keys']:
            restarted = self.queue.restart(key)
            if restarted:
                succeeded.append(str(key))
            else:
                failed.append(str(key))

        message = ''
        if len(succeeded) > 0:
            message += 'Restarted entries: {}.'.format(', '.join(succeeded))
            status = 'success'
        if len(failed) > 0:
            message += '\nNo finished entry for keys: {}'.format(', '.join(failed))
            status = 'error'

        answer = {'message': message.strip(), 'status': status}
        return answer