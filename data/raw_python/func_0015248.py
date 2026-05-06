def _strip_leading_dirname(self, path):
        '''Strip leading directory name from the given path'''
        return os.path.sep.join(path.split(os.path.sep)[1:])