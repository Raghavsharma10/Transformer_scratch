def read(self):
        '''Responses from an established socket'''
        responses = self._read()
        # Determine the number of messages in here and decrement our ready
        # count appropriately
        self.ready -= sum(
            map(int, (r.frame_type == Message.FRAME_TYPE for r in responses)))
        return responses