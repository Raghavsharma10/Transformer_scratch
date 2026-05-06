def identify(self, data):
        '''Send an identification message'''
        return self.send(constants.IDENTIFY, json.dumps(data))