def save(self, *args, **kwargs):
        """
        Saves this item.

        Creates a default base if there isn't
        one already.
        """
        with xact():
            if not self.vid:
                self.state = self.DRAFT

                if not self.object_id:
                    base = self._meta._base_model(is_published=False)
                    base.save(*args, **kwargs)
                    self.object = base

            super(VersionModel, self).save(*args, **kwargs)