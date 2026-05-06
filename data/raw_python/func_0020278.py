def query(self, model):
        '''Return a query for ``model`` when it needs to be indexed.
        '''
        session = self.router.session()
        fields = tuple((f.name for f in model._meta.scalarfields
                        if f.type == 'text'))
        qs = session.query(model).load_only(*fields)
        for related in self.get_related_fields(model):
            qs = qs.load_related(related)
        return qs