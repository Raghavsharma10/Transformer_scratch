def scope(self, key):
        '''Apply the name scope to a key

        Parameters
        ----------
        key : string

        Returns
        -------
        `name/key` if `name` is not `None`;
        otherwise, `key`.
        '''
        if self.name is None:
            return key
        return '{:s}/{:s}'.format(self.name, key)