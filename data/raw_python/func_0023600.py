def mpub(self, topic, *messages):
        '''Publish multiple messages to a topic'''
        return self.send(constants.MPUB + ' ' + topic, messages)