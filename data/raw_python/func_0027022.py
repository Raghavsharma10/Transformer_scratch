def clear(self, omit_item_evicted=False):
        """Empty the cache and optionally invoke item_evicted callback."""
        if not omit_item_evicted:
            items = self._dict.items()
            for key, value in items:
                self._evict_item(key, value)
        self._dict.clear()