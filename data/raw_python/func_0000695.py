def should_stale_item_be_fetched_synchronously(self, delta, *args, **kwargs):
        """
        Return whether to refresh an item synchronously when it is found in the
        cache but stale
        """
        if self.fetch_on_stale_threshold is None:
            return False
        return delta > (self.fetch_on_stale_threshold - self.lifetime)