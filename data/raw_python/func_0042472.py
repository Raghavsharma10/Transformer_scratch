def save(self, *args, **kwargs):
        """
        Takes an optional last_save keyword argument
        other wise last_save will be set to timezone.now()

        Calls super to actually save the object.
        """
        self.last_save = kwargs.pop('last_save', timezone.now())
        super(Cloneable, self).save(*args, **kwargs)