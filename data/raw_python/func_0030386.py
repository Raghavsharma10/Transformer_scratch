def new_transient(self, ext=''):
        '''Creates empty TransientFile with random name and given extension.
        File on FS is not created'''
        name = random_name(self.transient_length) + ext
        return TransientFile(self.transient_root, name, self)