def run(self):
        """Runs all errors, dependencies and run methods of all *Assistant objects in self.path.
        Raises:
            devassistant.exceptions.ExecutionException with a cause if something goes wrong
        """
        error = None
        # run 'pre_run', 'logging', 'dependencies' and 'run'
        try:  # serve as a central place for error logging
            self._logging(self.parsed_args)
            if 'deps_only' not in self.parsed_args:
                self._run_path_run('pre', self.parsed_args)
            self._run_path_dependencies(self.parsed_args)
            if 'deps_only' not in self.parsed_args:
                self._run_path_run('', self.parsed_args)
        except exceptions.ExecutionException as e:
            error = self._log_if_not_logged(e)
            if isinstance(e, exceptions.YamlError):  # if there's a yaml error, just shut down
                raise e

        # in any case, run post_run
        try:  # serve as a central place for error logging
            self._run_path_run('post', self.parsed_args)
        except exceptions.ExecutionException as e:
            error = self._log_if_not_logged(e)

        # exitfuncs are run all regardless of exceptions; if there is an exception in one
        #  of them, this function will raise it at the end
        try:
            utils.run_exitfuncs()
        except exceptions.ExecutionException as e:
            error = self._log_if_not_logged(e)

        if error:
            raise error