def fin(self):
        '''Indicate that this message is finished processing'''
        self.connection.fin(self.id)
        self.processed = True