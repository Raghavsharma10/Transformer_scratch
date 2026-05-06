def get_success_url(self):
        """Reverses the ``redis_metric_aggregate_detail`` URL using
        ``self.metric_slugs`` as an argument."""
        slugs = '+'.join(self.metric_slugs)
        url = reverse('redis_metric_aggregate_detail', args=[slugs])
        # Django 1.6 quotes reversed URLs, which changes + into %2B. We want
        # want to keep the + in the url (it's ok according to RFC 1738)
        # https://docs.djangoproject.com/en/1.6/releases/1.6/#quoting-in-reverse
        return url.replace("%2B", "+")