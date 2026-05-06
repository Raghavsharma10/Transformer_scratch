def sub(self, topic, channel):
        '''Subscribe to a topic/channel'''
        return self.send(' '.join((constants.SUB, topic, channel)))