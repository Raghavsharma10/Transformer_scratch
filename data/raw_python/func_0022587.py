def delete_metric(self, slug):
        """Removes all keys for the given ``slug``."""

        # To remove all keys for a slug, I need to retrieve them all from
        # the set of metric keys, This uses the redis "keys" command, which is
        # inefficient, but this shouldn't be used all that often.
        prefix = "m:{0}:*".format(slug)
        keys = self.r.keys(prefix)
        self.r.delete(*keys)  # Remove the metric data

        # Finally, remove the slug from the set
        self.r.srem(self._metric_slugs_key, slug)