def fetch_remaining_instances(self):
        """Read the derived table data for all objects tracked as remaining (=not found in the cache)."""
        if self.remaining_items:
            self.remaining_items = ContentItem.objects.get_real_instances(self.remaining_items)