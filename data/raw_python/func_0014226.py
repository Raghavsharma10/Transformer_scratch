def run(self):
        '''Performs the run through the templates and their providers'''
        for tpl in self.templates:
            for provider in tpl.providers:
                provider.provide()