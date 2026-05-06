def get_for_tag(self, tag):
        """
        Returns queryset of Entry instances assigned to specified
        tag, which can be a PK value, a slug value, or a Tag instance.

        :param tag: tag PK, slug, or instance.
        :rtype: django.db.models.query.QuerySet.
        """
        tag_filter = {'tag': tag}

        if isinstance(tag, six.integer_types):
            tag_filter = {'tag_id': tag}
        elif isinstance(tag, str):
            tag_filter = {'tag__slug': tag}

        return self.filter(id__in=[
            entry_tag.entry_id
            for entry_tag
            in EntryTag.objects.filter(**tag_filter)
        ])