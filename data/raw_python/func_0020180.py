def add_to_fields(self):
        '''Add this :class:`Field` to the fields of :attr:`model`.'''
        meta = self.model._meta
        meta.scalarfields.append(self)
        if self.index:
            meta.indices.append(self)