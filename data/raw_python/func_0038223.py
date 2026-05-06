def save(self, *args, **kwargs):
        """sets the `slug` values as the name

        :param args: inline arguments (optional)
        :param kwargs: keyword arguments (optional)
        :return: `super.save()`
        """
        self.slug = slugify(self.name)[:50]
        return super(Tag, self).save(*args, **kwargs)