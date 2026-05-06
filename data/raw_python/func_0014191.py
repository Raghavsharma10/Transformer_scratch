def iter_related(self):
        '''
        Generator function that iterates this object's related providers,
        which includes this provider.
        '''
        for tpl in self.provider_run.templates:
            yield tpl.providers[self.index]