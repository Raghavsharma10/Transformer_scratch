def _do_submit_problems(self):
        """Pull problems from the submission queue and submit them.

        Note:
            This method is always run inside of a daemon thread.
        """
        try:
            while True:
                # Pull as many problems as we can, block on the first one,
                # but once we have one problem, switch to non-blocking then
                # submit without blocking again.

                # `None` task is used to signal thread termination
                item = self._submission_queue.get()
                if item is None:
                    break

                ready_problems = [item]
                while len(ready_problems) < self._SUBMIT_BATCH_SIZE:
                    try:
                        ready_problems.append(self._submission_queue.get_nowait())
                    except queue.Empty:
                        break

                # Submit the problems
                _LOGGER.debug("Submitting %d problems", len(ready_problems))
                body = '[' + ','.join(mess.body for mess in ready_problems) + ']'
                try:
                    try:
                        response = self.session.post(posixpath.join(self.endpoint, 'problems/'), body)
                        localtime_of_response = epochnow()
                    except requests.exceptions.Timeout:
                        raise RequestTimeout

                    if response.status_code == 401:
                        raise SolverAuthenticationError()
                    response.raise_for_status()

                    message = response.json()
                    _LOGGER.debug("Finished submitting %d problems", len(ready_problems))
                except BaseException as exception:
                    _LOGGER.debug("Submit failed for %d problems", len(ready_problems))
                    if not isinstance(exception, SolverAuthenticationError):
                        exception = IOError(exception)

                    for mess in ready_problems:
                        mess.future._set_error(exception, sys.exc_info())
                        self._submission_queue.task_done()
                    continue

                # Pass on the information
                for submission, res in zip(ready_problems, message):
                    submission.future._set_clock_diff(response, localtime_of_response)
                    self._handle_problem_status(res, submission.future)
                    self._submission_queue.task_done()

                # this is equivalent to a yield to scheduler in other threading libraries
                time.sleep(0)

        except BaseException as err:
            _LOGGER.exception(err)