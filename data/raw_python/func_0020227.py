def load_data(self, data, callback=None):
        '''Load ``data`` from the :class:`stdnet.BackendDataServer`.'''
        return self.backend.execute(
            self.value_pickler.load_iterable(data, self.session), callback)