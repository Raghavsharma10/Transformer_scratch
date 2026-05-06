def _purge_config_file(self, key):
        """ Forget a previously deregister config file.  The caller should
        ensure that it was previously deregistered.
        """
        state = self.__load_state()
        del state['remove_configs'][key]
        self.__dump_state(state)