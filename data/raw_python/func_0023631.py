def stash(self, payload):
        """Stash the specified processes."""
        succeeded = []
        failed = []
        for key in payload['keys']:
            if self.queue.get(key) is not None:
                if self.queue[key]['status'] == 'queued':
                    self.queue[key]['status'] = 'stashed'
                    succeeded.append(str(key))
                else:
                    failed.append(str(key))
            else:
                failed.append(str(key))

        message = ''
        if len(succeeded) > 0:
            message += 'Stashed entries: {}.'.format(', '.join(succeeded))
            status = 'success'
        if len(failed) > 0:
            message += '\nNo queued entry for keys: {}'.format(', '.join(failed))
            status = 'error'

        answer = {'message': message.strip(), 'status': status}

        return answer