def clear(self, payload):
        """Clear queue from any `done` or `failed` entries.

        The log will be rotated once. Otherwise we would loose all logs from
        thoes finished processes.
        """
        self.logger.rotate(self.queue)
        self.queue.clear()
        self.logger.write(self.queue)

        answer = {'message': 'Finished entries have been removed.', 'status': 'success'}
        return answer