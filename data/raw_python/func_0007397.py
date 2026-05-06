def build(self, ignore=None):
        '''Calls all necessary methods to build the Lambda Package'''
        self._prepare_workspace()
        self.install_dependencies()
        self.package(ignore)