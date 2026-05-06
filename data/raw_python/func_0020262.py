def get(self, **kwargs):
        '''Return an instance of a model matching the query. A special case is
the query on ``id`` which provides a direct access to the :attr:`session`
instances. If the given primary key is present in the session, the object
is returned directly without performing any query.'''
        return self.filter(**kwargs).items(
            callback=self.model.get_unique_instance)