def save(self, *args, **kwargs):
        """creates the slug, queues up for indexing and saves the instance

        :param args: inline arguments (optional)
        :param kwargs: keyword arguments
        :return: `bulbs.content.Content`
        """
        if not self.slug:
            self.slug = slugify(self.build_slug())[:self._meta.get_field("slug").max_length]

        if not self.is_indexed:
            if kwargs is None:
                kwargs = {}
            kwargs["index"] = False
        content = super(Content, self).save(*args, **kwargs)
        index_content_contributions.delay(self.id)
        index_content_report_content_proxy.delay(self.id)
        post_to_instant_articles_api.delay(self.id)
        return content