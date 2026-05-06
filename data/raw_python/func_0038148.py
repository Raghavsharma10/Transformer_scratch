def save(self, *args, **kwargs):
        """Saving ensures that the slug, if not set, is set to the slugified name."""

        if not self.slug:
            self.slug = slugify(self.name)

        section = super(Section, self).save(*args, **kwargs)

        if self.query and self.query != {}:
            self._save_percolator()

        return section