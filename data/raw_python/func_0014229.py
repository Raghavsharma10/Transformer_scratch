def build_command(self):
        '''Returns the command to run, as a list (see subprocess module)'''
        # if defined in settings, run the function or return the string
        if self.options['command']:
            return self.options['command'](self) if callable(self.options['command']) else self.options['command']
        # build the default
        return self.build_default_command()