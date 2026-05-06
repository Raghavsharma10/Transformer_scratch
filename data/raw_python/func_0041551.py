def save(self, conflict_resolver=choose_mine):

        '''Save all options in memory to the `config_file`.

        Options are read once more from the file (to allow other writers to save configuration),
        keys in conflict are resolved, and the final results are written back to the file.

        :param conflict_resolver: a simple lambda or function to choose when an option key is
               provided from an outside source (THEIRS, usually a file on disk) but is also already
               set on this ConfigStruct (MINE)
        '''
        config = self._load(conflict_resolver)  # in case some other process has added items
        with open(self._config_file, 'wb') as cf:
            config.write(cf)