def on_failure(self, exc, task_id, args, kwargs, einfo):
        """on_failure

        http://docs.celeryproject.org/en/latest/userguide/tasks.html#task-inheritance

        :param exc: exception
        :param task_id: task id
        :param args: arguments passed into task
        :param kwargs: keyword arguments passed into task
        :param einfo: exception info
        """

        use_exc = str(exc)
        log.error(("{} FAIL - exc={} "
                   "args={} kwargs={}")
                  .format(
                     self.log_label,
                     use_exc,
                     args,
                     kwargs))