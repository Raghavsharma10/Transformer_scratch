def _do_load_results(self):
        """Submit a query asking for the results for a particular problem.

        To request the results of a problem: ``GET /problems/{problem_id}/``

        Note:
            This method is always run inside of a daemon thread.
        """
        try:
            while True:
                # Select a problem
                future = self._load_queue.get()
                # `None` task signifies thread termination
                if future is None:
                    break
                _LOGGER.debug("Loading results of: %s", future.id)

                # Submit the query
                query_string = 'problems/{}/'.format(future.id)
                try:
                    try:
                        response = self.session.get(posixpath.join(self.endpoint, query_string))
                    except requests.exceptions.Timeout:
                        raise RequestTimeout

                    if response.status_code == 401:
                        raise SolverAuthenticationError()
                    response.raise_for_status()

                    message = response.json()
                except BaseException as exception:
                    if not isinstance(exception, SolverAuthenticationError):
                        exception = IOError(exception)

                    future._set_error(IOError(exception), sys.exc_info())
                    continue

                # Dispatch the results, mark the task complete
                self._handle_problem_status(message, future)
                self._load_queue.task_done()

                # this is equivalent to a yield to scheduler in other threading libraries
                time.sleep(0)

        except Exception as err:
            _LOGGER.error('Load result error: ' + str(err))