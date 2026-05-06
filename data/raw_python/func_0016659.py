def _handle_problem_status(self, message, future):
        """Handle the results of a problem submission or results request.

        This method checks the status of the problem and puts it in the correct queue.

        Args:
            message (dict): Update message from the SAPI server wrt. this problem.
            future `Future`: future corresponding to the problem

        Note:
            This method is always run inside of a daemon thread.
        """
        try:
            _LOGGER.trace("Handling response: %r", message)
            _LOGGER.debug("Handling response for %s with status %s", message.get('id'), message.get('status'))

            # Handle errors in batch mode
            if 'error_code' in message and 'error_msg' in message:
                raise SolverFailureError(message['error_msg'])

            if 'status' not in message:
                raise InvalidAPIResponseError("'status' missing in problem description response")
            if 'id' not in message:
                raise InvalidAPIResponseError("'id' missing in problem description response")

            future.id = message['id']
            future.remote_status = status = message['status']

            # The future may not have the ID set yet
            with future._single_cancel_lock:
                # This handles the case where cancel has been called on a future
                # before that future received the problem id
                if future._cancel_requested:
                    if not future._cancel_sent and status == self.STATUS_PENDING:
                        # The problem has been canceled but the status says its still in queue
                        # try to cancel it
                        self._cancel(message['id'], future)
                    # If a cancel request could meaningfully be sent it has been now
                    future._cancel_sent = True

            if not future.time_received and message.get('submitted_on'):
                future.time_received = parse_datetime(message['submitted_on'])

            if not future.time_solved and message.get('solved_on'):
                future.time_solved = parse_datetime(message['solved_on'])

            if not future.eta_min and message.get('earliest_estimated_completion'):
                future.eta_min = parse_datetime(message['earliest_estimated_completion'])

            if not future.eta_max and message.get('latest_estimated_completion'):
                future.eta_max = parse_datetime(message['latest_estimated_completion'])

            if status == self.STATUS_COMPLETE:
                # TODO: find a better way to differentiate between
                # `completed-on-submit` and `completed-on-poll`.
                # Loading should happen only once, not every time when response
                # doesn't contain 'answer'.

                # If the message is complete, forward it to the future object
                if 'answer' in message:
                    future._set_message(message)
                # If the problem is complete, but we don't have the result data
                # put the problem in the queue for loading results.
                else:
                    self._load(future)
            elif status in self.ANY_STATUS_ONGOING:
                # If the response is pending add it to the queue.
                self._poll(future)
            elif status == self.STATUS_CANCELLED:
                # If canceled return error
                raise CanceledFutureError()
            else:
                # Return an error to the future object
                errmsg = message.get('error_message', 'An unknown error has occurred.')
                if 'solver is offline' in errmsg.lower():
                    raise SolverOfflineError(errmsg)
                else:
                    raise SolverFailureError(errmsg)

        except Exception as error:
            # If there were any unhandled errors we need to release the
            # lock in the future, otherwise deadlock occurs.
            future._set_error(error, sys.exc_info())