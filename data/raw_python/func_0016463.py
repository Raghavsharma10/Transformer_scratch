def qos(self, prefetch_size, prefetch_count, apply_global=False):
        """Request specific Quality of Service."""
        self.channel.basic_qos(prefetch_size, prefetch_count,
                                apply_global)