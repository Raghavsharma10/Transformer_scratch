def __load_state(self):
        """ Read persisted state from the JSON statefile
        """
        try:
            return ConfigState(json.load(open(self.config_state_path)))
        except (OSError, IOError) as exc:
            if exc.errno == errno.ENOENT:
                self.__dump_state({})
                return json.load(open(self.config_state_path))
            raise