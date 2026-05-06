def exec_module(self, module):
        '''import the source code, transforma it before executing it so that
           it is known to Python.'''
        global MAIN_MODULE_NAME
        if module.__name__ == MAIN_MODULE_NAME:
            module.__name__ = "__main__"
            MAIN_MODULE_NAME = None

        with open(self.filename) as f:
            source = f.read()

        if transforms.transformers:
            source = transforms.transform(source)
        else:
            for line in source.split('\n'):
                if transforms.FROM_EXPERIMENTAL.match(line):
                    ## transforms.transform will extract all such relevant
                    ## lines and add them all relevant transformers
                    source = transforms.transform(source)
                    break
        exec(source, vars(module))