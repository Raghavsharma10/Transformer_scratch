def _register_config_file(self, key, val):
        """ Persist a newly added config file, or update (overwrite) the value
        of a previously persisted config.
        """
        state = self.__load_state()
        if 'config_files' not in state:
            state['config_files'] = {}
        state['config_files'][key] = val
        self.__dump_state(state)