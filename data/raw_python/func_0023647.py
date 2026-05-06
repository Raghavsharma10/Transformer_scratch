def mpub(self, topic, messages, binary=True):
        '''Send multiple messages to a topic. Optionally pack the messages'''
        if binary:
            # Pack and ship the data
            return self.post('mpub', data=pack(messages)[4:],
                params={'topic': topic, 'binary': True})
        elif any('\n' in m for m in messages):
            # If any of the messages has a newline, then you must use the binary
            # calling format
            raise ClientException(
                'Use `binary` flag in mpub for messages with newlines')
        else:
            return self.post(
                '/mpub', params={'topic': topic}, data='\n'.join(messages))