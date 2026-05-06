def getrealpath(self, root, path):
        '''
        Return the real path on disk from the query path, from a root path.
        The input path from URL might be absolute '/abc', or point to parent '../test',
        or even with UNC or drive '\\test\abc', 'c:\test.abc',
        which creates security issues when accessing file contents with the path.
        With getrealpath, these paths cannot point to files beyond the root path.
        
        :param root: root path of disk files, any query is limited in root directory.
        
        :param path: query path from URL.
        '''
        if not isinstance(path, str):
            path = path.decode(self.encoding)
        # In windows, if the path starts with multiple / or \, the os.path library may consider it an UNC path
        # remove them; also replace \ to /
        path = pathrep.subn('/', path)[0]
        # The relative root is considered ROOT, eliminate any relative path like ../abc, which create security issues
        # We can use os.path.relpath(..., '/') but python2.6 os.path.relpath is buggy 
        path = os.path.normpath(os.path.join('/', path))
        # The normalized path can be an UNC path, or event a path with drive letter
        # Send bad request for these types
        if os.path.splitdrive(path)[0]:
            raise HttpInputException('Bad path')
        return os.path.join(root, path[1:])