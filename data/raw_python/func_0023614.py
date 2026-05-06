def pub(self, topic, message):
        '''Publish the provided message to the provided topic'''
        with self.random_connection() as client:
            client.pub(topic, message)
            return self.wait_response()