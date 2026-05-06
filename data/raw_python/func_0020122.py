def model(self, model, create=True):
        '''Returns the :class:`SessionModel` for ``model`` which
can be :class:`Model`, or a :class:`MetaClass`, or an instance
of :class:`Model`.'''
        manager = self.manager(model)
        sm = self._models.get(manager)
        if sm is None and create:
            sm = SessionModel(manager)
            self._models[manager] = sm
        return sm