def conf(self):
        '''Configuration (namedtuple)
        '''
        conf = namedtuple('conf', field_names=self._conf.keys())
        return conf(**self._conf)