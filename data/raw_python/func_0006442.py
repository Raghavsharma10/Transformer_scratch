def restore(self, name=None, keep=False, last=True, global_register=True, pixel_register=True):
        '''Restoring a configuration restore point.

        Parameters
        ----------
        name : str
            Name of the restore point. If not given, a md5 hash will be generated.
        keep : bool
            Keeping restore point for later use.
        last : bool
            If name is not given, the latest restore point will be taken.
        global_register : bool
            Restore global register.
        pixel_register : bool
            Restore pixel register.
        '''
        if name is None:
            if keep:
                name = next(reversed(self.config_state)) if last else next(iter(self.config_state))
                value = self.config_state[name]
            else:
                name, value = self.config_state.popitem(last=last)
        else:
            value = self.config_state[name]
            if not keep:
                value = copy.deepcopy(value)  # make a copy before deleting object
                del self.config_state[name]

        if global_register:
            self.global_registers = copy.deepcopy(value[0])
        if pixel_register:
            self.pixel_registers = copy.deepcopy(value[1])