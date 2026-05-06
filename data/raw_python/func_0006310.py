def default_run_conf(self):
        '''Default run configuration (namedtuple)
        '''
        default_run_conf = namedtuple('default_run_conf', field_names=self._default_run_conf.keys())
        return default_run_conf(**self._default_run_conf)