def apply_async(self, args=None, kwargs=None, **options):
        """ Do not run background task if previous task is uncompleted """
        if self.is_previous_task_processing(*args, **kwargs):
            message = 'Background task %s was not scheduled, because its predecessor is not completed yet.' % self.name
            logger.info(message)
            # It is expected by Celery that apply_async return AsyncResult, otherwise celerybeat dies
            return self.AsyncResult(options.get('task_id') or str(uuid4()))
        return super(BackgroundTask, self).apply_async(args=args, kwargs=kwargs, **options)