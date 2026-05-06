def save(self, *args, **kwargs):
        """Saving ensures that the slug, if not set, is set to the slugified name."""
        self.clean()

        if not self.slug:
            self.slug = slugify(self.name)

        super(SpecialCoverage, self).save(*args, **kwargs)

        if self.query and self.query != {}:
            # Always save and require client to filter active date range
            self._save_percolator()