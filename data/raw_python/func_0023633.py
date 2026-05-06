def remove(self, payload):
        """Remove specified entries from the queue."""
        succeeded = []
        failed = []
        for key in payload['keys']:
            running = self.process_handler.is_running(key)
            if not running:
                removed = self.queue.remove(key)
                if removed:
                    succeeded.append(str(key))
                else:
                    failed.append(str(key))
            else:
                failed.append(str(key))

        message = ''
        if len(succeeded) > 0:
            message += 'Removed entries: {}.'.format(', '.join(succeeded))
            status = 'success'
        if len(failed) > 0:
            message += '\nRunning or non-existing entry for keys: {}'.format(', '.join(failed))
            status = 'error'

        answer = {'message': message.strip(), 'status': status}

        return answer