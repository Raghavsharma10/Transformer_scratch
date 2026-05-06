def save(self, force_insert=False, **kwargs):
        """Send signals."""
        created = force_insert or not bool(self.pk)
        self.pre_save.send(self, created=created)
        super(Model, self).save(force_insert=force_insert, **kwargs)
        self.post_save.send(self, created=created)