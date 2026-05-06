def apply(self, *args: Any, **kwargs: Any) -> Any:
        """Called by workers to run the wrapped function.
        You may call it yourself if you want to run the task in current process
        without sending to the queue.

        If task has a `retry` property it will be retried on failure.

        If task has a `max_run_time` property the task will not be allowed to
        run more than that.
        """
        def send_signal(sig: Signal, **extra: Any) -> None:
            self._send_signal(sig, args=args, kwargs=kwargs, **extra)

        logger.debug("Applying %r, args=%r, kwargs=%r", self, args, kwargs)

        send_signal(signals.task_preapply)
        try:
            tries = 1 + self.retry
            while 1:
                tries -= 1
                send_signal(signals.task_prerun)
                try:
                    with time_limit(self.max_run_time or 0):
                        return self.f(*args, **kwargs)
                except Exception:
                    send_signal(signals.task_error, exc_info=sys.exc_info())
                    if tries <= 0:
                        raise
                else:
                    break
                finally:
                    send_signal(signals.task_postrun)
        except Exception:
            send_signal(signals.task_failure, exc_info=sys.exc_info())
            raise
        else:
            send_signal(signals.task_success)
        finally:
            send_signal(signals.task_postapply)