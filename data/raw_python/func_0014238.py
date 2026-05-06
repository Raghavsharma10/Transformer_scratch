def message(self, msg='', level=1, tab=0):
        '''Print a message to the console'''
        if self.verbosity >= level:
            self.stdout.write('{}{}'.format('    ' * tab, msg))