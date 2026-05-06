def find_file(self, filename: str, strip_path: bool = True, what='exposure') -> str:
        """Find file in the path"""
        if what == 'exposure':
            path = self._path
        elif what == 'header':
            path = self._headerpath
        elif what == 'mask':
            path = self._maskpath
        else:
            path = self._path
        tried = []
        if strip_path:
            filename = os.path.split(filename)[-1]
        for d in path:
            if os.path.exists(os.path.join(d, filename)):
                tried.append(os.path.join(d, filename))
                return os.path.join(d, filename)
        raise FileNotFoundError('Not found: {}. Tried: {}'.format(filename, ', '.join(tried)))