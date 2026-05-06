def mpub(self, topic, *messages):
        '''Publish messages to a topic'''
        with self.random_connection() as client:
            client.mpub(topic, *messages)
            return self.wait_response()