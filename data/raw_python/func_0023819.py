def exists(self, bucket, label):
        '''Whether a given bucket:label object already exists.'''
        fn = self._zf(bucket, label)
        try:
            self.z.getinfo(fn)
            return True
        except KeyError:
            return False