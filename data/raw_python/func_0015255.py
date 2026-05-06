def _report_problem(self, problem, level=logging.ERROR):
        '''Report a given problem'''
        problem = self.basename + ': ' + problem
        if self._logger.isEnabledFor(level):
            self._problematic = True
        if self._check_raises:
            raise DapInvalid(problem)
        self._logger.log(level, problem)