def getAll(self):
        '''Return a dictionary with all variables'''

        if not bool(len(self.ATTRIBUTES)):
            self.load_attributes()
        return eval(str(self.ATTRIBUTES))