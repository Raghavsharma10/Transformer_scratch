def schedule_task(self):
        """
        Schedules this publish action as a Celery task.
        """
        from .tasks import publish_task

        publish_task.apply_async(kwargs={'pk': self.pk}, eta=self.scheduled_time)