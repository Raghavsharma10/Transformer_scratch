def run_conf(self):
        '''Run configuration (namedtuple)
        '''
        run_conf = namedtuple('run_conf', field_names=self._run_conf.keys())
        return run_conf(**self._run_conf)