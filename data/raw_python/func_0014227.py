def write(self, content):
        '''Provider instances use this to write to the buffer'''
        self.buffer.write(content)
        if settings.DEBUG:
            self.buffer.write('\n')