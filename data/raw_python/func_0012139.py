def parent(self):
        """return the parent URL, with params, query, and fragment in place"""
        path = '/'.join(self.path.split('/')[:-1])
        s = path.strip('/').split(':')
        if len(s)==2 and s[1]=='':
            return None
        else:
            return self.__class__(self, path=path)