def save(self, *args, **kwargs):
        """sets the `slug` values as the name

        :param args: inline arguments (optional)
        :param kwargs: keyword arguments (optional)
        :return: `super.save()`
        """
        if self.slug is None or self.slug == "":
            self.slug = slugify(self.name)

        feature_type = super(FeatureType, self).save(*args, **kwargs)

        if self.instant_article_is_dirty:
            index_feature_type_content.delay(self.pk)

        self._db_instant_article = self.instant_article

        # Run all behaviors for `create`
        if self.is_new:
            update_feature_type_rates.delay(self.pk)

        return feature_type