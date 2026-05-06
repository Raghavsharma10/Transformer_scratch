def load_library(self, path):
        '''
        Load a template library into the state.
        '''
        module = import_module(path)
        self.tags.update(module.register.tags)
        self.helpers.update(module.register.helpers)