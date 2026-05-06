def send(self, instance, *args, **kwargs):
        """Send signal."""
        for receiver in self.receivers:
            receiver(instance, *args, **kwargs)