def query(self, model, **kwargs):
        '''Create a new :class:`Query` for *model*.'''
        sm = self.model(model)
        query_class = sm.manager.query_class or Query
        return query_class(sm._meta, self, **kwargs)