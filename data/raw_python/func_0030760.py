def check(self, field):
        '''
        Returns permissions determined by object itself
        '''
        if self.permissions is None:
            return field.parent.permissions
        return self.permissions