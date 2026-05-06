def pipe_to_process(self, payload):
        """Send something to stdin of a specific process."""
        message = payload['input']
        key = payload['key']
        if not self.process_handler.is_running(key):
            return {'message': 'No running process for this key',
                    'status': 'error'}
        self.process_handler.send_to_process(message, key)
        return {'message': 'Message sent',
                'status': 'success'}