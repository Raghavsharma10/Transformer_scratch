def switch(self, payload):
        """Switch the two specified entry positions in the queue."""
        first = payload['first']
        second = payload['second']
        running = self.process_handler.is_running(first) or self.process_handler.is_running(second)
        if running:
            answer = {
                'message': "Can't switch running processes, "
                "please stop the processes before switching them.",
                'status': 'error'
            }

        else:
            switched = self.queue.switch(first, second)
            if switched:
                answer = {
                    'message': 'Entries #{} and #{} switched'.format(first, second),
                    'status': 'success'
                }
            else:
                answer = {'message': "One or both entries do not exist or are not queued/stashed.",
                          'status': 'error'}
        return answer