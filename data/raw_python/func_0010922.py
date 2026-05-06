def get_updates(self, **kwargs):
        '''Get parameter update expressions for performing optimization.

        Keyword arguments can be applied here to set any of the global
        optimizer attributes.

        Yields
        ------
        updates : (parameter, expression) tuples
            A sequence of parameter updates to be applied during optimization.
        '''
        self._prepare(**kwargs)
        for param, grad in self._differentiate():
            for var, update in self._get_updates_for(param, grad):
                # For auxiliary variables, updates are meant to replace the
                # existing variable value.
                if var != param:
                    yield var, update
                    continue
                # If momentum is disabled, just apply the parameter delta.
                if self.momentum == 0:
                    yield var, param - update
                    continue
                # Momentum is enabled, so we keep track of velocity here.
                vel_tm1 = util.shared_like(param, 'vel')
                vel_t = util.as_float(self.momentum) * vel_tm1 - update
                if self.nesterov:
                    # see http://arxiv.org/pdf/1212.0901v2.pdf (eq 7) and
                    # https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617
                    mom_sqr = util.as_float(self.momentum ** 2)
                    mom_inc = util.as_float(1 + self.momentum)
                    vel_t = mom_sqr * vel_tm1 - mom_inc * update
                yield vel_tm1, vel_t
                yield param, param + vel_t