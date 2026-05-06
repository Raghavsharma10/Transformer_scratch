def send_status(self, payload):
        """Send the daemon status and the current queue for displaying."""
        answer = {}
        data = []
        # Get daemon status
        if self.paused:
            answer['status'] = 'paused'
        else:
            answer['status'] = 'running'

        # Add current queue or a message, that queue is empty
        if len(self.queue) > 0:
            data = deepcopy(self.queue.queue)
            # Remove stderr and stdout output for transfer
            # Some outputs are way to big for the socket buffer
            # and this is not needed by the client
            for key, item in data.items():
                if 'stderr' in item:
                    del item['stderr']
                if 'stdout' in item:
                    del item['stdout']
        else:
            data = 'Queue is empty'
        answer['data'] = data

        return answer