def _initialize_manager(self, runtime):
        """Sets the runtime, configuration and json client"""
        if self._runtime is not None:
            raise errors.IllegalState('this manager has already been initialized.')
        self._runtime = runtime
        self._config = runtime.get_configuration()
        set_json_client(runtime)