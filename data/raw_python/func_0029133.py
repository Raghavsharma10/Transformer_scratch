def on_success(self, retval, task_id, args, kwargs):
        """on_success

        http://docs.celeryproject.org/en/latest/reference/celery.app.task.html

        :param retval: return value
        :param task_id: celery task id
        :param args: arguments passed into task
        :param kwargs: keyword arguments passed into task
        """

        log.info(("{} SUCCESS - retval={} task_id={} "
                  "args={} kwargs={}")
                 .format(
                     self.log_label,
                     retval,
                     task_id,
                     args,
                     kwargs))