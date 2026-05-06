def object(self, session):
        '''Instance of :attr:`model_type` with id :attr:`object_id`.'''
        if not hasattr(self, '_object'):
            pkname = self.model_type._meta.pkname()
            query = session.query(self.model_type).filter(**{pkname:
                                                             self.object_id})
            return query.items(callback=self.__set_object)
        else:
            return self._object