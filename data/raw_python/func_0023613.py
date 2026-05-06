def wait_response(self):
        '''Wait for a response'''
        responses = self.read()
        while not responses:
            responses = self.read()
        return responses