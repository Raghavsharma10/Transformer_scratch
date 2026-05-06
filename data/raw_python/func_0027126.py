def save(self, *args, **kwargs):
        """
        Updates next_trigger_at field if:
         - instance become active
         - instance.schedule changed
         - instance is new
        """
        try:
            prev_instance = self.__class__.objects.get(pk=self.pk)
        except self.DoesNotExist:
            prev_instance = None

        if prev_instance is None or (not prev_instance.is_active and self.is_active or
                                     self.schedule != prev_instance.schedule):
            self.update_next_trigger_at()

        super(ScheduleMixin, self).save(*args, **kwargs)