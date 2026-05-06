def get_dependencies(self, version=None):
        '''
        Parameters
        ----------
        version: str
            string representing version number whose dependencies you are
            looking up
        '''

        version = _process_version(self, version)
        history = self.get_history()

        for v in reversed(history):
            if BumpableVersion(v['version']) == version:
                return v['dependencies']

        raise ValueError('Version {} not found'.format(version))