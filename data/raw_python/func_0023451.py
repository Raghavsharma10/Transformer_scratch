def find(cls, name):
        '''Find the exception class by name'''
        if not cls.mapping:  # pragma: no branch
            for _, obj in inspect.getmembers(exceptions):
                if inspect.isclass(obj):
                    if issubclass(obj, exceptions.NSQException):  # pragma: no branch
                        if hasattr(obj, 'name'):
                            cls.mapping[obj.name] = obj
        klass = cls.mapping.get(name)
        if klass == None:
            raise TypeError('No matching exception for %s' % name)
        return klass