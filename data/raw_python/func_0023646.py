def pub(self, topic, message):
        '''Publish a message to a topic'''
        return self.post('pub', params={'topic': topic}, data=message)