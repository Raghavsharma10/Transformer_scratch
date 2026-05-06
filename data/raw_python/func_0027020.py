def compute(self):
        """Computes the value. Does not look at the cache."""
        self.value = self.value_provider()
        if self.value is not_computed:
            return None
        else:
            return self.value