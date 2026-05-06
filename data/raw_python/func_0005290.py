def setup(self, app):
        '''
        Setup properties from parent app on the command
        '''
        self.logger = app.logger
        self.shell.logger = self.logger

        if not self.command_name:
            raise EmptyCommandNameException()

        self.app = app
        self.arguments_declaration = self.arguments
        self.arguments = app.arguments

        if self.use_subconfig:
            _init_config(self)
        else:
            self.config = self.app.config