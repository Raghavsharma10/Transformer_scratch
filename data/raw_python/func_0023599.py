def pub(self, topic, message):
        '''Publish to a topic'''
        return self.send(' '.join((constants.PUB, topic)), message)