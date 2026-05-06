def storage_method(func):
    '''Calls decorated method with VersionedStorage as self'''
    def wrap(self, *args, **kwargs):
        return func(self._root_storage, *args, **kwargs)
    return wrap