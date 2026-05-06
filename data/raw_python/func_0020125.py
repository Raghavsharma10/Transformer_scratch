def new(self, *args, **kwargs):
        '''Create a new instance of :attr:`model` and commit it to the backend
server. This a shortcut method for the more verbose::

    instance = manager.session().add(MyModel(**kwargs))
'''
        return self.session().add(self.model(*args, **kwargs))