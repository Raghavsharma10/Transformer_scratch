def _prepare(self, **kwargs):
        '''Set up properties for optimization.

        This method can be overridden by base classes to provide parameters that
        are specific to a particular optimization technique (e.g., setting up a
        learning rate value).
        '''
        self.learning_rate = util.as_float(kwargs.pop('learning_rate', 1e-4))
        self.momentum = kwargs.pop('momentum', 0)
        self.nesterov = kwargs.pop('nesterov', False)
        self.patience = kwargs.get('patience', 5)
        self.validate_every = kwargs.pop('validate_every', 10)
        self.min_improvement = kwargs.pop('min_improvement', 0)
        self.max_gradient_norm = kwargs.pop('max_gradient_norm', 0)
        self.max_gradient_elem = kwargs.pop('max_gradient_elem', 0)

        util.log_param('patience', self.patience)
        util.log_param('validate_every', self.validate_every)
        util.log_param('min_improvement', self.min_improvement)
        util.log_param('max_gradient_norm', self.max_gradient_norm)
        util.log_param('max_gradient_elem', self.max_gradient_elem)
        util.log_param('learning_rate', self.learning_rate)
        util.log_param('momentum', self.momentum)
        util.log_param('nesterov', self.nesterov)