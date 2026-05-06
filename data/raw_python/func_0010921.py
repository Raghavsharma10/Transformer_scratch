def _compile(self, **kwargs):
        '''Compile the Theano functions for evaluating and updating our model.
        '''
        util.log('compiling evaluation function')
        self.f_eval = theano.function(self._inputs,
                                      self._monitor_exprs,
                                      updates=self._updates,
                                      name='evaluation')
        label = self.__class__.__name__
        util.log('compiling {} optimizer'.format(click.style(label, fg='red')))
        updates = list(self._updates) + list(self.get_updates(**kwargs))
        self.f_step = theano.function(self._inputs,
                                      self._monitor_exprs,
                                      updates=updates,
                                      name=label)