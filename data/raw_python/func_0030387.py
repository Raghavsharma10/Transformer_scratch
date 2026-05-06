def get_transient(self, name):
        '''Restores TransientFile object with given name.
        Should be used when form is submitted with file name and no file'''
        # security checks: basically no folders are allowed
        assert not ('/' in name or '\\' in name or name[0] in '.~')
        transient = TransientFile(self.transient_root, name, self)
        if not os.path.isfile(transient.path):
            raise OSError(errno.ENOENT, 'Transient file has been lost',
                          transient.path)
        return transient