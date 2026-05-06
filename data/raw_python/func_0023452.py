def exception(self):
        '''Return an instance of the corresponding exception'''
        code, _, message = self.data.partition(' ')
        return self.find(code)(message)