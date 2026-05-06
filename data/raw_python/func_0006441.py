def create_restore_point(self, name=None):
        '''Creating a configuration restore point.

        Parameters
        ----------
        name : str
            Name of the restore point. If not given, a md5 hash will be generated.
        '''
        if name is None:
            for i in iter(int, 1):
                name = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f') + '_' + str(i)
                try:
                    self.config_state[name]
                except KeyError:
                    break
                else:
                    pass
        if name in self.config_state:
            raise ValueError('Restore point %s already exists' % name)
        self.config_state[name] = (copy.deepcopy(self.global_registers), copy.deepcopy(self.pixel_registers))
        return name