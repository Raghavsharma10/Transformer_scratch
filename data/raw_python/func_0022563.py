def _mem(self):
        """Record Memory usage."""
        value = int(psutil.virtual_memory().percent)
        set_metric("memory", value, category=self.category)
        gauge("memory", value)