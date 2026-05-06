def _skip_lines(self, n):
        '''Skip a number of lines from the output.'''
        for i in range(n):
            self.line = next(self.output)
        return self.line