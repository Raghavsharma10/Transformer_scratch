def check_if_ready(self):
        """Check for and fetch the results if ready."""
        try:
            results = self.manager.check(self.results_id)
        except exceptions.ResultsNotReady as e:
            self._is_ready = False
            self._not_ready_exception = e
        except exceptions.ResultsExpired as e:
            self._is_ready = True
            self._expired_exception = e
        else:
            failures = self.get_failed_requests(results)
            members = self.get_new_members(results)
            self.results = self.__class__.Results(list(members), list(failures))
            self._is_ready = True
            self._not_ready_exception = None