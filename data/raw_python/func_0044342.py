def _deregister_config_file(self, key):
        """ Deregister a previously registered config file.  The caller should
        ensure that it was previously registered.
        """
        state = self.__load_state()
        if 'remove_configs' not in state:
            state['remove_configs'] = {}
        state['remove_configs'][key] = (state['config_files'].pop(key))
        self.__dump_state(state)